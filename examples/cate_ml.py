from esteem import drivers, parallel
from esteem.wrappers import amber, orca, mace
import glob

# List solutes and solvents and get default arguments
all_solutes = {'cate': 'catechol'}
all_solvents = {'cycl': 'cyclohexane', 'meth': 'methanol'}
all_solutes.update(all_solvents)

from esteem.tasks.solutes import SolutesTask
from esteem.tasks.solvate import SolvateTask
from esteem.tasks.clusters import ClustersTask
from esteem.tasks.spectra import SpectraTask
from esteem.tasks.ml_training import MLTrainingTask

solutes_task = SolutesTask()
solvate_task = SolvateTask()
clusters_task = ClustersTask()
spectra_task = SpectraTask()
mltrain_task = MLTrainingTask()

# Some simple overrides for a quick job
solutes_task.basis = '6-31G'
solutes_task.func = 'PBE'
solutes_task.directory = 'gs_PBE'
solutes_task.wrapper = orca.ORCAWrapper()
solutes_task.script_settings = parallel.get_default_script_settings(solutes_task.wrapper)
solutes_task.wrapper.setup(nprocs=8)
solutes_task.ref_mol_dir = "gs_PBE"

solvate_task.boxsize = 18
solvate_task.ewaldcut = 10
solvate_task.nsnaps = 20
solvate_task.md_geom_prefix = 'gs_PBE/'
solvate_task.wrapper = amber.AmberWrapper()
solvate_task.script_settings = parallel.get_default_script_settings(solvate_task.wrapper)

clusters_task.nroots = 1
clusters_task.radius = 2
clusters_task.output = 'orca'
clusters_task.basis = solutes_task.basis
clusters_task.func = solutes_task.func
clusters_task.ref_mol_dir = None
clusters_task.wrapper = orca.ORCAWrapper()
clusters_task.wrapper.setup(nprocs=8)
clusters_task.script_settings = parallel.get_default_script_settings(clusters_task.wrapper)

mltrain_task.wrapper = mace.MACEWrapper()
iter_dir_suffixes = []
seeds=["{solu}_{solv}"]
traj_suffixes = ["orca"] # what trajectory suffixes to train on
dir_suffixes = {"orca": "exc"} # what directory suffixes to append to the seeds to find each trajectory suffix in
ntraj = {("gs","orca"):1,("es1","orca"):0} # how many trajectories of each suffix to expect, labelled A, B, C etc
targets = {0: "gs",1: "es1"} # which targets to train for
mltrain_task.wrapper.train_args['max_num_epochs'] = 500
mltrain_task.wrapper.train_args['restart_latest'] = True
from esteem.active_learning import create_mltrain_tasks
mltrain_tasks = create_mltrain_tasks(mltrain_task,train_calcs=["MACE"],seeds=seeds,targets=targets,rand_seed={'a':123},meth="",truth="orca",traj_suffixes=traj_suffixes,dir_suffixes=dir_suffixes,ntraj=ntraj,iter_dir_suffixes=[],delta_epochs=500,separate_valid=False)

spectra_task.inputformat = clusters_task.wrapper
spectra_task.output = 'test.png'
spectra_task.warp_scheme = None
spectra_task.wavelength = (200,400,1)
spectra_task.broad = 0.2
spectra_task.files = glob.glob(f'orca/*es1*.out')

# Run main driver
drivers.main(all_solutes,all_solvents,
             solutes_task,solvate_task,clusters_task,spectra_task,all_mltrain_tasks=mltrain_tasks,
             all_qmd_tasks=solutes_task,
             make_script=parallel.make_sbatch)
exit()

