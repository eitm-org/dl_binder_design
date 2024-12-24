#!/usr/bin/env python

import os, sys

from pyrosetta import *
from pyrosetta.rosetta import *

import numpy as np
from collections import OrderedDict
import time
import argparse
import subprocess
import time
import glob

import torch
import json

import util_protein_mpnn as mpnn_util
from Bio import SeqIO
from omegaconf import OmegaConf

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent, 'include'))
from silent_tools import silent_tools

init( "-beta_nov16 -in:file:silent_struct_type binary -mute all" +
    " -use_terminal_residues true -mute basic.io.database core.scoring" )

def cmd(command, wait=True):
    the_command = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if (not wait):
        return
    the_stuff = the_command.communicate()
    return str( the_stuff[0]) + str(the_stuff[1] )

def range1( iterable ): return range( 1, iterable + 1 )

class sample_features():
    '''
    This is a struct which keeps all the features related to a single sample together
    '''

    def __init__(self, pose, tag):
        self.pose = pose
        self.tag = os.path.basename(tag).split('.')[0]
    
    def parse_fixed_res(self):
        '''
            Parse which residues are fixed from the residue remarks
        '''

        # Iterate over the residue positions in this pose and check if they are fixed
        fixed_list = []

        endA = self.pose.split_by_chain()[1].total_residue()
        for resi in range1( endA ):
            reslabels = self.pose.pdb_info().get_reslabels( resi )

            if len(reslabels) == 0: continue

            if str(self.pose.pdb_info().get_reslabels( resi )[1]).strip() == 'FIXED':
                fixed_list.append( resi )

        # Get ordered unique chainIDs for the pose
        # The last chain (B) is the target and will have a fixed sequence
        self.chains = list( OrderedDict.fromkeys( [ self.pose.pdb_info().chain(i) for i in range1( self.pose.total_residue() ) ] ) )

        # Create the fixed res dict, this will be input to ProteinMPNN
        self.fixed_res = {
            self.chains[0]: fixed_list,
            self.chains[1]: []
        }
    
    def thread_mpnn_seq(self, binder_seq):
        '''
        Thread the binder sequence onto the pose being designed
        '''
        rsd_set = self.pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

        for resi, mut_to in enumerate( binder_seq ):
            resi += 1 # 1 indexing
            name3 = mpnn_util.aa_1_3[ mut_to ]
            new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
            self.pose.replace_residue( resi, new_res, True )
    

class ProteinMPNN_runner():
    '''
    This class is designed to run the ProteinMPNN model on a single input. This class handles
    the loading of the model, the loading of the input data, the FastRelax cycles, and the processing of the output
    '''

    def __init__(self, args, struct_manager):
        self.struct_manager = struct_manager

        if torch.cuda.is_available():
            print('Found GPU will run ProteinMPNN on GPU')
            self.device = "cuda:0"
        else:
            print('No GPU found, running ProteinMPNN on CPU')
            self.device = "cpu"

        # Configs for the FastRelax cycles
        xml = os.path.join(script_dir, 'RosettaFastRelaxUtil.xml')
        objs = protocols.rosetta_scripts.XmlObjects.create_from_file(xml)

        self.FastRelax = objs.get_mover('FastRelax')

        self.relax_cycles = args.relax_cycles

    def relax_pose(self, sample_feats):
        '''
        Run FastRelax on the current pose
        '''

        relaxT0 = time.time()

        print('Running FastRelax')

        self.FastRelax.apply(sample_feats.pose)

        print(f"Completed one cycle of FastRelax in {int(time.time()) - relaxT0} seconds")

    def proteinmpnn(self, sequences, sample_feats):
        '''
        Run ProteinMPNN sequence optimization on the pose, this does not use FastRelax
        '''

        # Iterate though each seq score pair and thread the sequence onto the pose
        # Then write each pose to a pdb file
        outtag = f"{sample_feats.tag}_dldesign"

        for s in sequences: 
            for seq_record in SeqIO.parse(s, "fasta"):
                seq = seq_record.seq

            sample_feats.thread_mpnn_seq(seq)
            self.struct_manager.dump_pose(sample_feats.pose, outtag)

    def proteinmpnn_fastrelax(self, sequence, sample_feats):
        '''
        Run ProteinMPNN plus FastRelax on the pose being designed
        '''
        tot_t0 = time.time()
        design_counter = 0

        outtag = f"{sample_feats.tag}_dldesign"

        for cycle in range(args.relax_cycles):

            for seq_record in SeqIO.parse(sequence, "fasta"):
                seq = seq_record.seq
            
            sample_feats.thread_mpnn_seq(seq)

            self.relax_pose(sample_feats)

            if args.output_intermediates:
                tag = f"{outtag}_0_cycle{cycle}"
                self.struct_manager.dump_pose(sample_feats.pose, tag)

        sample_feats.thread_mpnn_seq(seq)
        tag = f"{outtag}_cycle{args.relax_cycles}"
        self.struct_manager.dump_pose(sample_feats.pose, tag)

    def run_model(self, pdb, seq, args):
        t0 = time.time()

        print(f"Attempting pose: {pdb}")
        
        # Load the pose 
        pose = self.struct_manager.load_pose(pdb)

        # Initialize the features
        sample_feats = sample_features(pose, pdb)

        # Parse the fixed residues from the pose remarks
        # sample_feats.parse_fixed_res()

        # Now determine which type of run we are doing and execute it
        if args.relax_cycles > 0:
            if args.seqs_per_struct > 1:
                raise Exception('Cannot use --seqs_per_struct > 1 with --relax_cycles > 0')

            self.proteinmpnn_fastrelax(seq, sample_feats)

        else:
            self.proteinmpnn(sample_feats)

        seconds = int(time.time() - t0)

        print( f"Struct: {pdb} reported success in {seconds} seconds" )

class StructManager():
    '''
    This class handles all of the input and output for the ProteinMPNN model. It deals with silent files vs. pdbs,
    checkpointing, and writing of outputs

    Note: This class could be moved to a separate file
    '''

    def __init__(self, args):
        self.args = args
        self.silent = False
        self.pdb = False

        if not args.pdbdir == None:
            self.pdb = True
            self.seq = True

            self.pdbdir    = args.pdbdir
            self.outdir = args.outdir

            self.struct_iterator = glob.glob(os.path.join(args.pdbdir, '*.pdb'))

        self.silent = False
        if not args.silent == '':
            self.silent = True

            self.struct_iterator = silent_tools.get_silent_index(args.silent)['tags']

            self.sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
            self.sfd_in.read_file(args.silent)

            self.sfd_out = core.io.silent.SilentFileData(args.outdir, False, False, "binary", core.io.silent.SilentFileOptions())
            self.outdir = args.outdir

        # Assert that either silent or pdb is true, but not both
        assert(self.silent ^ self.pdb), f'Both silent and pdb are set to {args.silent} and {args.pdb} respectively. Only one of these may be active at a time'

        # Setup checkpointing
        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

    def record_checkpoint(self, tag):
        '''
        Record the fact that this tag has been processed.
        Write this tag to the list of finished structs
        '''
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\n')

    def iterate(self):
        '''
        Iterate over the silent file or pdb directory and run the model on each structure
        '''

        # Iterate over the structs and for each, check that the struct has not already been processed
        for struct in self.struct_iterator:
            tag = os.path.basename(struct).split('.')[0]
            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue

            yield struct

    def dump_pose(self, pose, tag):
        '''
        Dump this pose to either a silent file or a pdb file depending on the input arguments
        '''
        # If the outdir does not exist, create it
        # If there are parents in the path that do not exist, create them as well
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        if self.pdb:
            pdbfile = os.path.join(self.outdir, tag + '.pdb')
            pose.dump_pdb(pdbfile)
        
        if self.silent:
            struct = self.sfd_out.create_SilentStructOP()
            struct.fill_struct(pose, tag)

            self.sfd_out.add_structure(struct)
            self.sfd_out.write_silent_struct(struct, self.outdir)

    def load_pose(self, tag):
        '''
        Load a pose from either a silent file or a pdb file depending on the input arguments
        '''

        if not self.pdb and not self.silent:
            raise Exception('Neither pdb nor silent is set to True. Cannot load pose')

        if self.pdb:
            pose = pose_from_pdb(tag)
        
        if self.silent:
            pose = Pose()
            self.sfd_in.get_structure(tag).fill_pose(pose)
        
        return pose


####################
####### Main #######
####################

def run(config_file):

    # Read configuration file
    conf = OmegaConf.load(config_file)

    ##### Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument( "-seqdir", type=str, default=conf.af2ig.seqdir, help='The name of a directory of seqs to run through the model' )
    parser.add_argument( "-pdbdir", type=str, default=conf.af2ig.pdbdir, help='The name of a directory of pdbs to run through the model' )
    parser.add_argument( "-silent", type=str, default=conf.af2ig.silent, help='The name of a silent file to run through the model' )
    parser.add_argument( "-outdir", type=str, default=os.path.join(conf.af2ig.outdir, 'fast_relax'), help='The directory to which all output files will be written' )

    parser.add_argument( "-checkpoint_name", type=str, default=os.path.join(conf.af2ig.outdir, 'check.point'), help="The name of a file where tags which have finished will be written (default: check.point)" )
    parser.add_argument( "-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')

    # Design Arguments
    parser.add_argument( "-relax_cycles", type=int, default=1, help="The number of relax cycles to perform on each structure (default: 1)" )
    parser.add_argument( "-output_intermediates", action="store_true", help='Whether to write all intermediate sequences from the relax cycles to disk (default: False)' )
    parser.add_argument( "-seqs_per_struct", type=int, default="1", help="The number of sequences to generate for each structure (default: 1)" )

    args = parser.parse_args( sys.argv[1:] )

    struct_manager     = StructManager(args)
    proteinmpnn_runner = ProteinMPNN_runner(args, struct_manager)

    for design in struct_manager.iterate():
        pdb = design
        seq = os.path.join(args.seqdir, os.path.basename(pdb).split('.')[0] + '.fa')

        if args.debug: proteinmpnn_runner.run_model(pdb, seq, args)

        else: # When not in debug mode the script will continue to run even when some poses fail
            t0 = time.time()

            try: proteinmpnn_runner.run_model(pdb, seq, args)

            except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

            except:
                seconds = int(time.time() - t0)
                print( "Struct with tag %s failed in %i seconds with error: %s"%( design, seconds, sys.exc_info()[0] ) )

        # We are done with one pdb, record that we finished
        struct_manager.record_checkpoint(design)
    