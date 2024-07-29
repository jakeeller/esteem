from copy import deepcopy

# Import the drivers and wrappers we need
from esteem import drivers, parallel
from esteem.wrappers.orca import ORCAWrapper
from esteem.wrappers.amber import AmberWrapper
from esteem.wrappers.mace import MACEWrapper

# Import the task classes we need
from esteem.tasks.solutes import SolutesTask
from esteem.tasks.solvate import SolvateTask
from esteem.tasks.clusters import ClustersTask
from esteem.tasks.spectra import SpectraTask
from esteem.tasks.ml_training import MLTrainingTask
from esteem.tasks.ml_testing import MLTestingTask
from esteem.tasks.ml_trajectories import MLTrajTask

# Setup solute and solvents and target states
all_solutes = {'cate': 'catechol'}
all_solvents = {'cycl': 'cyclohexane', 'meth': 'methanol'}
all_solutes.update(all_solvents)
targets = {'gs':0,'es1':1,'es2':2}
orcacmd="/storage/nanosim/orca6/orca_6_0_0_shared_openmpi416_avx2/orca"

# Set up tasks
solutes_task = SolutesTask()
solvate_task = SolvateTask()
clusters_task = ClustersTask()
spectra_task = SpectraTask()
mltrain_task = MLTrainingTask()
mltest_task = MLTestingTask()
mltraj_task = MLTrajTask()

solutes_task.wrapper = ORCAWrapper()
solutes_task.wrapper.setup(nprocs=8)
from ase.calculators.orca import OrcaProfile
solutes_task.script_settings = parallel.get_default_script_settings(solutes_task.wrapper)
solutes_task.wrapper.orcaprofile=OrcaProfile(command=orcacmd)
# Setup different XC functionals as different tasks
all_solutes_tasks = {}
funcs = ['PBE','PBE0','wB97M-D3BJ']
basis_sets = ['def2-TZVPPD']
for basis in basis_sets:
    for func in funcs:
        for target in targets:  
            prefix = f'{target}_{func}'
            solutes_task.disp = True if 'D3BJ' not in func else False
            all_solutes_tasks[prefix] = deepcopy(solutes_task)
            all_solutes_tasks[prefix].func = func
            all_solutes_tasks[prefix].basis = basis
            all_solutes_tasks[prefix].target = targets[target]
            all_solutes_tasks[prefix].directory = prefix

# Set up solvate task
solvate_task.wrapper = AmberWrapper()
solvate_task.script_settings = parallel.get_default_script_settings(solvate_task.wrapper)
solvate_task.boxsize = 15
solvate_task.ewaldcut = 9.0
all_solvate_tasks = {'md': solvate_task}

# Set up clusters task
clusters_task.wrapper = ORCAWrapper()
clusters_task.script_settings = parallel.get_default_script_settings(clusters_task.wrapper)
clusters_task.wrapper.setup(nprocs=8,maxcore=2500)
clusters_task.wrapper.orcaprofile=OrcaProfile(command=orcacmd)
clusters_task.output = 'orca'
clusters_task.nroots = 2
all_clusters_tasks = {}

# specific radii for combinations of solvent and solute
# to ensure equal-sized clusters
solv_rad = {}
for rad in [2.5,5.0]:
    solv_rad[rad] = {'cate_meth': rad,
                     'cate_cycl': rad,
                     'meth_meth': rad+1.5,
                     'cycl_cycl': rad+0.5}
    # Set up task as per size above
    for traj in ['A','B']:
        clusters_task.max_atoms = 185
        clusters_task.max_snapshots = 90 if traj=='A' else 100
        clusters_task.min_snapshots = 0 if traj=='A' else 90
        clusters_task.radius = solv_rad[rad]
        clusters_task.which_traj = traj
        suffix = f'solvR{rad}_{traj}'
        clusters_task.exc_suffix = suffix
        all_clusters_tasks[suffix] = deepcopy(clusters_task)

# Now create tasks for clusters runs for each Active Learning iteration
meth="mace"
truth="orca"
train_calcs = ["MACEac0u","MACEac1u","MACEac2u"]
#train_calcs = ["MACEr"]
seed="{solu}_{solv}"
traj_suffix = "orca"
md_suffix = "mldyn"
md_dir_suffix = 'mldyn'
rand_seed = {'a':123,'b':456}
from esteem.active_learning import create_clusters_tasks
active_clusters_tasks = create_clusters_tasks(clusters_task,train_calcs,seed,traj_suffix,md_suffix,
                          md_dir_suffix,targets,rand_seed,meth,truth)
all_clusters_tasks.update(active_clusters_tasks)

# Create tasks for ML Training
mltrain_task.wrapper = MACEWrapper()
iter_dir_suffixes = []
seeds=["{solu}_{solv}","{solv}_{solv}"] # format of the seeds
traj_suffixes = ["orca"] # what trajectory suffixes to train on
dir_suffixes = {"orca": "solvR2.5"} # what directory suffixes to append to the seeds to find each trajectory suffix in
ntraj = {("gs","orca"):1,("es1","orca"):0} # how many trajectories of each suffix to expect, labelled A, B, C etc
targets = {0: "gs",1: "es1"} # which targets to train for
mltrain_task.wrapper.train_args['max_num_epochs'] = 500
iter_dir_suffixes = ["mltraj"]
from esteem.active_learning import create_mltrain_tasks
mltrain_task.ntraj=270
mltrain_tasks = create_mltrain_tasks(mltrain_task,train_calcs=train_calcs,
                                     seeds=seeds,targets=targets,rand_seed=rand_seed,
                                     meth="",truth="orca",traj_suffixes=traj_suffixes,
                                     dir_suffixes=dir_suffixes,ntraj=ntraj,
                                     iter_dir_suffixes=iter_dir_suffixes,
                                     delta_epochs=500,separate_valid=True)

from esteem.active_learning import create_mltraj_tasks
mltraj_task.wrapper = MACEWrapper()
mltraj_task.snap_wrapper = MACEWrapper()
mltraj_task.geom_prefix = 'gs_PBE'
mltraj_task.calc_seed = "{solu}_{solv}"
mltraj_task.md_init_traj_link = f"{{solu}}_{{solv}}_exc/{{solu}}_{{solv}}_gs_A_carved.traj"
mltraj_tasks = create_mltraj_tasks(mltraj_task,train_calcs=train_calcs,targets=targets,
                    rand_seed=rand_seed,meth="",traj_suffix='mldyn',
                    md_wrapper=mltraj_task.wrapper,snap_wrapper=mltraj_task.snap_wrapper,
                    two_targets=True)

spectra_task.inputformat = "traj"
spectra_task.output = 'test_ml.png'
spectra_task.warp_scheme = None
spectra_task.wavelength = (200,400,1)
spectra_task.broad = 0.2
spectra_task.trajectory = [[f"{{solu}}_{{solv}}_gs_MACE_mldyn/{{solu}}_{{solv}}_gs_A_MACEa_mldyn.traj"]]
all_spectra_tasks = {'default': spectra_task}

# Invoke main driver
drivers.main(all_solutes,all_solvents,
             all_solutes_tasks=all_solutes_tasks,
             all_solvate_tasks=all_solvate_tasks,
             all_clusters_tasks=all_clusters_tasks,
             all_mltrain_tasks=mltrain_tasks,
             all_qmd_tasks=solutes_task,
             all_mltraj_tasks=mltraj_tasks,
             make_script=parallel.make_sbatch)
# Quit - function defs for interactive use might follow
exit()

