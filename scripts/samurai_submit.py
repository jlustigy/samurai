"""
"""

import os
import datetime
import shutil
import stat

def create_run_directory(reldir="runs/"):
    """
    """

    now = datetime.datetime.now()
    startstr = now.strftime("%Y-%m-%d--%H-%M")

    dstdir = os.path.join(reldir, startstr)

    os.mkdir(dstdir)

    file1 = "samurai_run.py"
    dst1 = os.path.join(dstdir, file1)
    shutil.copy(file1, dst1)

    return

def generate_job_script(pyfile="samurai_run.py", jobname="jobscript.job",
                        workdir="/gscratch/vsm/jlustigy/csh/random_crap"):
    """
    """

    newfile = os.path.join(workdir, jobname)

    f = open(newfile, 'w')
    f.write('ipython %s' %pyfile)
    f.close()

    st = os.stat(newfile)
    os.chmod(newfile, st.st_mode | stat.S_IEXEC)

    return

def generate_pbs_script(name="samurai", pbsname="samurai.csh",
                        oedir="/gscratch/vsm/jlustigy/csh/random_crap",
                        workdir="/gscratch/vsm/jlustigy/csh/random_crap",
                        runfile="jobscript*.job",
                        hyak_slots=4,
                        walltime='200:00:00',
                        nodes=1, ppn=12, mem="24gb", feature="12core"):
    """
    """
    f = open(os.path.join(workdir, pbsname), 'w')

    f.write('#!/bin/bash\n')
    f.write('## NAME FOR YOUR JOB\n')
    f.write('#PBS -N %s\n' %name)
    f.write('\n')
    f.write('## SPECIFY NODE DETAILS\n')
    f.write('#PBS -l nodes=%i:ppn=%i,mem=%s,feature=%s\n' %(nodes, ppn, mem, feature))
    f.write('\n')
    f.write('#PBS -l walltime=%s\n' %walltime)
    f.write('\n')
    f.write('## Put the STDOUT and STDERR from jobs into the below directory\n')
    f.write('#PBS -o %s\n' %oedir)
    f.write('## Put both the stderr and stdout into a single file\n')
    f.write('#PBS -j oe\n')
    f.write('\n')
    f.write('## Sepcify the working directory for this job bundle\n')
    f.write('#PBS -d %s\n' %workdir)
    f.write('\n')

    if hyak_slots is not None:
        f.write("# If you can't run as many tasks as there are cores due to memory constraints\n")
        f.write("# you can simply set HYAK_SLOTS to a number instead.\n")
        f.write('HYAK_SLOTS=%i\n' %hyak_slots)

    f.write('\n')

    f.write('### Debugging information\n')
    f.write('### Include your job logs which contain output from the below commands\n')
    f.write('###  in any job-related help requests.\n')
    f.write('# Total Number of processors (cores) to be used by the job\n')
    f.write('HYAK_NPE=$(wc -l < $PBS_NODEFILE)\n')
    f.write('# Number of nodes used\n')
    f.write('HYAK_NNODES=$(uniq $PBS_NODEFILE | wc -l )\n')
    f.write('echo "**** Job Debugging Information ****"\n')
    f.write('echo "This job will run on $HYAK_NPE total CPUs on $HYAK_NNODES different nodes"\n')
    f.write('echo ""\n')
    f.write('echo "ENVIRONMENT VARIABLES"\n')
    f.write('set\n')
    f.write('echo "**********************************************"\n')
    f.write('### End Debugging information\n')

    f.write('\n')

    f.write('ulimit -s unlimited\n')

    f.write('\n')

    f.write('find . -name "%s" | parallel -j $HYAK_SLOTS\n' %runfile)

    f.write('\n')

    f.write('exit 0\n')

    f.close()

    return

def submit_job():
    """
    """

    return

if __name__ == "__main__":

    name = "cg_30m_crescent_map"

    runname = "samurai_"+name
    rundir = os.path.join("/gscratch/vsm/jlustigy/mapping/runs/", name)
    pyfile = "samurai_run.py"
    jobname="jobscript.job"
    pbsname = "samurai.csh"

    generate_job_script(workdir=rundir, pyfile=pyfile, jobname=jobname)

    generate_pbs_script(name=runname, pbsname=pbsname, oedir=rundir, workdir=rundir, runfile=jobname)
