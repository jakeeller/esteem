#!/usr/bin/env python
# coding: utf-8

"""Defines the MACEWrapper Class"""
import numpy as np
from ase.io import read,write

class MACEWrapper():
    """
    Sets up, trains and runs a MACE Neural Network Calculator to represent a
    potential energy surface.
    """

    from mace import tools
    default_train_args = vars(tools.build_default_arg_parser().parse_args(["--name","dummy","--train_file","train.xyz"]))
    
    def __init__(self,**kwargs):
        """Sets up instance attributes for MACEWrapper """
        from copy import deepcopy
        self.train_args = deepcopy(self.default_train_args)
        self.train_args['max_num_epochs'] = -1
        self.train_args['model'] = 'EnergyDipolesMACE'
        self.train_args['loss'] = 'energy_forces_dipole'
        self.train_args['error_table'] = 'EnergyDipoleRMSE'
        self.train_args['restart_latest'] = True

        # Allow overrides for this instance of the class
        for kw in self.train_args:
            if kw in kwargs:
                self.train_args[kw] = kwargs[kw]

        # Make a set of default loading arguments by copying in training arguments
        self.load_args = {}
        self.calc_ext = ""
        self.log_ext = ""
        self.calc = None
        self.calc_params = None
        self.atom_e = 0.0
        self.atom_energies = {}
        self.atoms_on_load = None
        self.output = "mace"

    def calc_filename(self,seed,target,prefix='',suffix=''):
        if target is None or target == 0:
            calcfn = seed+"_gs_"+suffix
        else:
            calcfn = seed+"_es"+str(target)+"_"+suffix
            
        calcfn = prefix + calcfn
            
        return calcfn
    
    def load(self,seed,target=None,prefix="",suffix="",dir_suffix=""):
        """
        Loads an existing MACE Calculator

        seed: str
        
        target: int

        suffix: str

        kwargs: dict
            other keywords
        """

        # Check against or store previous calculator parameters
        #if self.calc_params is not None:
        #    if ((self.calc_params['target'] != target) or
        #        (self.calc_params['calc_suffix'] != suffix) or
        #        (self.calc_params['calc_seed'] != seed)):
        #        raise Exception('Attempted change of calculator parameters for previously-loaded wrapper. Not supported.')
        if self.calc_params is None:
            self.calc_params = {'target': target,'calc_prefix': prefix,
                                'calc_suffix': suffix,'calc_seed': seed}

        if self.calc is not None: 
            return self.calc
        from mace.calculators import MACECalculator
        from os import path

        # Find checkpoint file(s) for calculator
        calctarget = self.calc_params['target']
        if isinstance(suffix,dict) or isinstance(target,list):
            self.calc = []
        suffixes = suffix if isinstance(suffix,dict) else {suffix:self.train_args['seed']}
        targets = calctarget if isinstance(calctarget,list) else [calctarget]
        for suff in suffixes:
            for targ in targets:
                dirname = self.calc_filename(seed,targ,prefix=prefix,suffix=dir_suffix)
                calcfn = self.calc_filename(seed,targ,prefix="",suffix=suff)
                modelfile = f"{dirname}/{calcfn}_swa.model"
                if path.exists(modelfile):
                    checkpoint = modelfile
                else:
                    modelfile = f"{dirname}/{calcfn}.model"
                    if path.exists(modelfile):
                        checkpoint = modelfile
                    else:
                        checkpoints_dir=f"{dirname}/checkpoints"
                        checkpoint = f"{checkpoints_dir}/{calcfn}_run-{suffixes[suff]}.model"
                print(f'# Loading Calculator from: {checkpoint} with args: {self.load_args}',flush=True)
                calc = MACECalculator(checkpoint,device="cuda",default_dtype='float64',model_type='EnergyDipoleMACE')
                if isinstance(suffix,dict) or isinstance(calctarget,list):
                    self.calc.append(calc)
                else:
                    self.calc = calc

        return self.calc

    def traj_to_extxyz(self,trajfile,outfilename):

        from ase.io import Trajectory
        from esteem import trajectories

        traj = Trajectory(trajfile)
        
        from ase.io.extxyz import write_xyz
        f=open(outfilename,"w")
        write_xyz(f,traj)
        f.close()

        return outfilename, len(traj)
   
    def reset_loss(self,seed,prefix="",suffix="",target=None):
        """
        Runs training for MACE model using an input trajectory as training points

        seed: str
        
        target: int

        suffix: str
        
        prefix: str
        
        """

        from os import path

        # Not sensible to try training in other directory than current, so prefix is
        # suppressed here but used elsewhere (eg for retrieving trajs)
        label = self.calc_filename(seed,target,prefix="",suffix=suffix)
        
    def train(self,seed,prefix="",suffix="",dir_suffix="",trajfile="",validfile=None,testfile=None,
              target=None,restart=False,**kwargs):
        """
        Runs training for MACE model using an input trajectory as training points

        seed: str
        
        target: int

        suffix: str

        trajfile: str

        restart: bool

        kwargs: dict

        """
        
        from os import path

        # Not sensible to try training in other directory than current, so prefix is
        # suppressed here but used elsewhere (eg for retrieving trajs)
        dirname = self.calc_filename(seed,target,prefix="",suffix=dir_suffix)

        # Sort out optional arguments to see if any overrides to defaults have been supplied
        from copy import deepcopy
        import os
        train_args = deepcopy(self.train_args)
        for kw in train_args:
            if kw in kwargs:
                train_args[kw] = kwargs[kw]

        # Check if training already complete
        calcfn = self.calc_filename(seed,target,prefix="",suffix=suffix)
        finished_model = None
        modelfile = f"{dirname}/{calcfn}_swa.model"
        if path.exists(modelfile):
            finished_model = modelfile
        else:
            modelfile = f"{dirname}/{calcfn}.model"
            if path.exists(modelfile):
                finished_model = modelfile
            else:
                checkpoints_dir=f"{dirname}/checkpoints"
                modelfile = f"{checkpoints_dir}/{calcfn}_run-{train_args['seed']}.model"
                if path.exists(modelfile):
                    finished_model = modelfile
        if finished_model is not None:
            print(f'# Skipping training as finished model {finished_model} already exists')
            print(f'# Delete this if training is intended to be restarted / extended')
            return

        # Convert trajectories from .traj to .xyz format
        convtrajfile = {'train':trajfile}
        if validfile is not None:
            convtrajfile['valid'] = validfile
        else: # assume we want to use a validation fraction of 5%
            del train_args['valid_file']
            train_args['valid_fraction'] = 0.05
        if testfile is not None:
            convtrajfile['test'] = testfile
        for key in convtrajfile:
            trajf = convtrajfile[key]
            print(f'# Converting trajectory {trajf} to extxyz format as {key} file')
            extxyzfile = self.calc_filename(seed,target,prefix=dirname+"/",suffix=suffix)+f"_{key}.xyz"
            extxyzfile, ntraj = self.traj_to_extxyz(trajf,extxyzfile)
            print(f'# Wrote {ntraj} frames to {extxyzfile} in extxyz format')
            extxyzfile = self.calc_filename(seed,target,prefix="",suffix=suffix)+f"_{key}.xyz"
            train_args[f'{key}_file'] = extxyzfile
            if key=='train':
                if 'test' not in convtrajfile:
                    train_args['test_file'] = extxyzfile
                if self.train_args['max_num_epochs']<0:
                    train_args['max_num_epochs'] = int(round(200000*train_args['batch_size']/ntraj/50)*50)
        
        # Open atom_traj
        from ase.io import Trajectory
        atom_traj = Trajectory(f'{seed}_atoms_{suffix}.traj')
        
        # Now switch working directory to ensure all outputs are together
        orig_dir = os.getcwd()
        os.chdir(dirname)

        # Set up input data
        train_args['name'] = self.calc_filename(seed,target,prefix="",suffix=suffix)
        train_args['device'] = 'cuda'

        # Calculate E0s from atom_traj
        from esteem.trajectories import atom_energies
        from ase.data import atomic_numbers
        atom_en = atom_energies(atom_traj)
        atom_traj.close()
        E0s = {}
        for sym in atom_en:
            E0s[atomic_numbers[sym]] = atom_en[sym]
        train_args['E0s'] = E0s
        
        # Some fixes to the input parameter list that prevent breakage
        if train_args['start_swa'] is None:
            train_args['start_swa'] = train_args['max_num_epochs'] // 4 * 3
        for arg in ['num_channels','max_L']:
            if arg in train_args:
                if train_args[arg] is None:
                    del train_args[arg]
        for arg in ['wandb_project','wandb_entity','wandb_name']:
            if arg in train_args:
                if train_args[arg]=="":
                    del train_args[arg]
        if 'wandb_log_hypers' in train_args:
            del train_args['wandb_log_hypers']
        for arg in train_args:
            if train_args[arg] is False:
                print(arg)
        for arg in ['save_cpu','restart_latest','keep_checkpoints','ema','swa','amsgrad',
                    'wandb','mean','std','distributed','save_all_checkpoints',
                    'foundation_model','foundation_model_readout','pair_repulsion',
                    'statistics_file','atomic_numbers','compute_polarizability']:
            if arg in train_args:
                if train_args[arg] is not True:
                    del train_args[arg]
                else:
                    train_args[arg] = ""

        # Write config.txt
        import sys
        store_argv = sys.argv
        config_file = self.calc_filename(seed,target,prefix="",suffix=suffix)+'_config.txt'
        print(f'# Writing MACE configuration to {config_file}')
        with open(config_file,"w") as f:
            for kw in train_args:
                eq = '=' if train_args[kw]!="" else ""
                f.write(f'--{kw}{eq}{train_args[kw]}\n')
        #sys.argv = ['train.py',config_file]
        
        sys.argv = ['run_train.py']
        for kw in train_args:
            sys.argv.append(f'--{kw}');
            if train_args[kw]!="":
                sys.argv.append(f'{train_args[kw]}')
        
        extxyzfile = train_args[f'train_file']
        print(f'# Training MACE model using trajectory {extxyzfile} with parameters:')
        print('#',train_args)
        import scripts.run_train
        scripts.run_train.main()
        sys.argv = store_argv
        
        os.chdir(orig_dir)
    
    def traj_write(self,atoms,traj):
        kw = {#'dipole': atoms.get_dipole_moment(),
              #'charges': atoms.get_charges(),
              'energy': atoms.get_potential_energy(),
              'forces': atoms.get_forces()}
        traj.write(atoms,**kw)

    def restore_from_coordinates(self,model,crdfile):
        model = read(crdfile,index=-1)
        return model

    def heatup(self,model,seed,calc_params={},nsteps=100):
        """
        Runs a heating up Molecular Dynamics calculation with the MACE ASE calculator.

        model: ASE Atoms

        seed: str
        
        calc_params: dict
        
        nsteps: int
        """

        if not hasattr(self,'dynamics') or self.dynamics is None:
            from types import SimpleNamespace
            self.dynamics = SimpleNamespace()
            self.dynamics.type = "LANG"
            self.dynamics.friction = self.friction
            self.dynamics.new_traj = True
        nramp = 10
        models_ramped = []
        for iramp in range(nramp):
            ramp_temp = self.temp0*(iramp+1)/nramp
            self.dynamics = self.run_md(model,seed+"_heat",calc_params,nsteps/nramp,self.dt,1,ramp_temp,dynamics=self.dynamics)
            models_ramped.append(model.copy())
            write('heat.xyz',models_ramped)

    def densityequil(self,model,seed,calc_params={},nsteps=100):
        """
        Runs a density equilibration Molecular Dynamics calculation with the MACE ASE calculator.
        model: ASE Atoms
        seed: str
        calc_params: dict
        nsteps: int
        """

        if ((not hasattr(self,'dynamics') or self.dynamics is None) or
            (self.dynamics.type!="NPT")):
            from types import SimpleNamespace
            self.dynamics = SimpleNamespace()
            self.dynamics.type = "NPT"
            self.dynamics.friction = self.friction
            self.dynamics.ttime = self.ttime
            from ase import units
            self.dynamics.pfactor = 2*1.06e9*(units.J/units.m**3)*self.ttime**2
            self.dynamics.new_traj = True
        self.dynamics = self.run_md(model,seed+"_dens",calc_params,nsteps,self.dt,1,self.temp0,dynamics=self.dynamics)
        write('density.xyz',model)

    def equil(self,model,seed,calc_params={},nsteps=100):
        """
        Runs an equilibration Molecular Dynamics calculation with the MACE ASE calculator.
        model: ASE Atoms
        seed: str
        calc_params: dict
        nsteps: int
        """

        if ((not hasattr(self,'dynamics') or self.dynamics is None) or
            (self.dynamics.type!="LANG")):
            from types import SimpleNamespace
            self.dynamics = SimpleNamespace()
            self.dynamics.type = "LANG"
            self.dynamics.friction = self.friction
            self.dynamics.new_traj = True
        self.dynamics = self.run_md(model,seed+"_equil",calc_params,nsteps,self.dt,1,self.temp0,dynamics=self.dynamics)
        write('equil.xyz',model)

    def snapshots(self,model,seed,calc_params={},nsnaps=1,nsteps=100,start=0,nat_solu=0,nat_solv=0):
        """
        Runs a density equilibration Molecular Dynamics calculation with the MACE ASE calculator.
        model: ASE Atoms
        seed: str
        calc_params: dict
        nsteps: int
        """
        from ase.io import Trajectory

        trajname = seed+'.traj'
        if start==0:
            traj = Trajectory(trajname, 'w')
        else:
            traj = Trajectory(trajname)
            prevframe = len(traj)
            print(f'# Found trajectory file {trajname} containing {prevframe} frames.')
            if prevframe != start+1:
                raise Exception(f'Error: This does not agree with the resumption point {start}.')
            traj.close()
            traj = Trajectory(trajname, 'a')
        if True:
            #((not hasattr(self,'dynamics') or self.dynamics is None) or
            #(self.dynamics.type!="LANG")):
            from types import SimpleNamespace
            self.dynamics = SimpleNamespace()
            self.dynamics.type = "LANG"
            self.dynamics.friction = self.friction
            self.dynamics.new_traj = True
        snapout = model.copy()
        for step in range(start,nsnaps):
            self.dynamics = self.run_md(snapout,seed+"_snaps",calc_params,nsteps,self.dt,1,self.temp0,dynamics=self.dynamics)
            from esteem.tasks.clusters import reimage_cluster
            snap_reimage = snapout.copy()
            reimage_cluster(snap_reimage,nat_solv,nat_solu)
            traj.write(snap_reimage)
            write(f'snap{step:04}.xyz',snap_reimage)

    def run_md(self,model,mdseed,calc_params,md_steps,md_timestep,superstep,temp,
                 solvent=None,charge=0,restart=False,readonly=False,constraints=None,dynamics=None,
                 continuation=None):
        """
        Runs a Molecular Dynamics calculation with the MACE ASE calculator.

        model: ASE Atoms

        seed: str
        
        calc_params: dict
        
        md_steps: int

        md_timestep: float

        superstep: int

        temp: float

        target: int or None

        solvent: str or None

        restart: bool

        readonly: bool
        """

        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md import Langevin, npt
        from ase.io import Trajectory
        from ase import units

        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        dir_suffix = calc_params['calc_dir_suffix']
        prefix = calc_params['calc_prefix']

        # Load the MACE Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix,dir_suffix=dir_suffix)
        model.calc = calc_ml

        # Initialise velocities if this is first step, otherwise inherit from model
        if np.all(model.get_momenta() == 0.0):
            MaxwellBoltzmannDistribution(model,temperature_K=temp)

        # For each ML superstep, remove C.O.M. translation and rotation    
        #Stationary(model)
        #ZeroRotation(model)
        #print(f'constraints: {model.constraints}')

        if readonly:
            model = read(mdseed+".xyz") # Read final image
            model.calc = calc_ml
            model.get_potential_energy() # Recalculate energy for final image
            return None
        else:
            new_traj = dynamics.new_traj
            friction = dynamics.friction
            dyn_type = dynamics.type
            if dyn_type=="LANG":
                if type(dynamics)!=Langevin:
                    dynamics = Langevin(model, timestep=md_timestep, temperature_K=temp, friction=friction)
                else: # in case they have changed
                    dynamics.set_timestep(md_timestep)
                    dynamics.set_friction(friction)
                    dynamics.set_temperature(temperature_K=temp)
            if dyn_type=="NPT" and type(dynamics)!=npt.NPT:
                ttime = dynamics.ttime
                pfactor = dynamics.pfactor #None #1.06e9*(units.J/units.m**3)*ttime**2 # Bulk modulus for ethanol
                dynamics = npt.NPT(model, timestep=md_timestep, temperature_K=temp, externalstress=0,
                                   pfactor=pfactor,ttime=ttime,mask=np.eye(3,dtype=bool))
            dynamics.new_traj = new_traj
            dynamics.friction = friction
            dynamics.type = dyn_type
            if new_traj:
                if hasattr(dynamics,'traj'):
                    dynamics.traj.close()
                    dynamics.observers = []
                dynamics.traj = Trajectory(mdseed+".traj", 'w', model)
                dynamics.attach(self.traj_write, interval=1, atoms=model, traj=dynamics.traj)
                dynamics.new_traj = False
            
            dynamics.run(md_steps)
            return dynamics


    # Define a MACE geometry optimisation function
    def geom_opt(self,model,seed,calc_params,driver_tol='default',
                 solvent=None,charge=0,spin=0,writeonly=False,readonly=False,continuation=False,cleanup=False,
                 traj=None):
        """
        Runs a geometry optimisation calculation with the MACE ASE calculator

        model: ASE Atoms

        seed: str
        
        calc_params: dict

        dummy: str

        driver_tol:

        target: int or None

        solvent: str or None

        readonly: bool
        """
        from ase.io import Trajectory
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        dir_suffix = calc_params['calc_dir_suffix']
        prefix = calc_params['calc_prefix']
        
        from ase.optimize import BFGS
        from ase.units import Hartree, Bohr

        # Load the appropriate MACE Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix,dir_suffix=dir_suffix)
        model.calc = calc_ml

        # Create instance of BFGS optimizer, run it and return results
        dyn = BFGS(model,trajectory=traj)

        # tolerances corresponding to NWChem settings
        fac=1
        if driver_tol=='default':
            fmax = 0.00045*fac
        if driver_tol=='loose':
            fmax = 0.00450*fac
        if driver_tol=='veryloose':
            fmax = 0.1*fac
        if driver_tol=='tight':
            fmax = 0.000015*fac
            
        dyn.run(fmax=fmax)

        return model.get_potential_energy(), model.get_forces(), model.get_positions()

    def freq(self,model_opt,seed,calc_params,solvent=None,charge=0,
             temp=300,writeonly=False,readonly=False,continuation=False,cleanup=True):
        """
        Runs a Vibrational Frequency calculation with the MACE ASE calculator
        
        model_opt: ASE Atoms

        seed: str
        
        suffix: str

        dummy: str

        driver_tol:

        target: int or None

        solvent: str or None
        
        temp: float

        readonly: bool
        """

        from ase.vibrations import Vibrations, Infrared
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        dir_suffix = calc_params['calc_dir_suffix']
        prefix = calc_params['calc_prefix']

        # Load the appropriate MACE  Calculator
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix,dir_suffix=dir_suffix)
        model_opt.calc = calc_ml

        # Create instance of Vibrations class, run it and return results
        #vib = Vibrations(model_opt,name=self.calc_filename(seed,target,prefix=prefix,suffix=suffix))
        #vib.run()
        #freqs = vib.get_frequencies()
        #vib.summary()
        #vib.clean()
        ir = Infrared(model_opt,name=self.calc_filename(seed,target,prefix=prefix,suffix=suffix))
        ir.run()
        ir.summary()
        ir.write_spectra(out=ir.name+'_ir_spectrum.dat',start=0,end=4000,width=20)
        ir.clean()
        
        #print(freqs)
        return ir

    def singlepoint(self,model,seed,calc_params,solvent=None,charge=0,spin=0,
                    forces=False,dipole=False,calc=False,readonly=False,continuation=False,
                    cleanup=True):
        """
        Runs a singlepoint calculation with the MACE ASE calculator

        model: ASE Atoms

        seed: str
        
        suffix: str

        dummy: str

        target: int or None

        solvent: str or None

        readonly: bool
        """
        
        calc_seed = calc_params['calc_seed']
        target = calc_params['target']
        suffix = calc_params['calc_suffix']
        dir_suffix = calc_params['calc_dir_suffix']
        prefix = calc_params['calc_prefix']
        
        # Load the appropriate MACE Calculator(s)
        calc_ml = self.load(calc_seed,target,prefix=prefix,suffix=suffix,dir_suffix=dir_suffix)
        if isinstance(calc_ml,list):
            e_calc = []
            f_calc = []
            d_calc = []
            for calc in calc_ml:
                calc.results = {}
                model.calc = calc
                e_calc.append(model.get_potential_energy())
                if forces:
                    f_calc.append(model.get_forces())
                if dipole:
                    d_calc.append(model.get_dipole_moment())
            e_calc = np.array(e_calc)
            model.calc.results["energy"] = e_calc
            model.calc.results["energy_std"] = np.std(e_calc)
            if dipole:
                d_calc = np.array(d_calc)
                model.calc.results["dipole"] = d_calc
                model.calc.results["dipole_std"] = np.std(d_calc,axis=1)
            if forces:
                f_calc = np.array(f_calc)
                model.calc.results["forces"] = f_calc
                model.calc.results["forces_std"] = np.std(f_calc,axis=1)
        else:
            model.calc = calc_ml
            e_calc = model.get_potential_energy()
            if forces:
                f_calc = model.get_forces()
            if dipole:
                d_calc = model.get_dipole_moment()
        res = [e_calc]
        if forces:
            res.append(f_calc)
        if dipole:
            res.append(d_calc)
        if calc:
            res.append(calc_ml)
        return res



