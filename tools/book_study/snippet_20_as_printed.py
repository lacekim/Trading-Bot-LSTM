"""AFML Chapter 20 multiprocessing engine, Snippets 20.5-20.14 (PDF pages 189-199), copied from the PDF with only the fixes
the user ruled on 2026-11-17. (20.1-20.4 are separate files: snippet_20_1/_2/_3/_4_as_printed.py.)

Every change from the printed text:
  | snippet | printed                                        | here                                                   |
  | 20.6/7/13 | xrange                                       | range                                                  |
  | 20.7    | ’’’ around the docstring (and ’molecule’ in it) | ''' (the quotes inside the docstring are left as text) |
  | 20.7    | argList (undefined; NameError)                 | pdObj (the argument's name)                            |
  | 20.7    | jobs=[]    for i in xrange(...):  (one line)   | two lines                                              |
  | 20.7    | for i in out:df0=df0.append(i)                 | for i in out:df0=pd.concat([df0,i])                    |
  | 20.9    | time, sys, dt used, imports not printed        | import time,sys / import datetime as dt                 |
  | 20.11   | copy_reg; method.im_func / im_self / im_class  | copyreg; method.__func__ / __self__ / type(method.__self__) |
  | 20.12   | ’’’ around the docstring; copy, pd not imported | '''; import copy / import pandas as pd                 |
  | 20.14   | the usage line printed above getPCs            | kept as a comment: it is used as printed where fileNames, path, eVec exist (3.4 rule) |
Ligatures in names (ﬂoat, ﬁleNames) are written as plain letters; Python reads them identically.
Not changed: pd.Series() without dtype in 20.7 (object dtype under pandas 2+; values unchanged).
"""
import time,sys                      # 20.9: used, not printed
import datetime as dt                # 20.9: used, not printed (the book's name)
import copy                          # 20.12: used, not printed
import pandas as pd                  # 20.12: used, not printed
# SNIPPET 20.5
import numpy as np
#———————————————————————————————————————
def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts
# SNIPPET 20.6
def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1 + 4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are the heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts
# SNIPPET 20.7
def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a DataFrame or Series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kargs: any other argument needed by func
    Example: df1=mpPandasObj(func,(’molecule’,df0.index),24,**kargs)
    '''
    import pandas as pd
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)
    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else:out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=pd.concat([df0,i])
    df0=df0.sort_index()
    return df0
# SNIPPET 20.8
def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out
# SNIPPET 20.9
import multiprocessing as mp
#———————————————————————————————————————
def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs,(time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return
#———————————————————————————————————————
def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a ’func’ callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asynchronous output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out
# SNIPPET 20.10
def expandCall(kargs):
    # Expand the arguments of a callback function, kargs[’func’]
    func=kargs['func']
    del kargs['func']
    out=func(** kargs)
    return out
# SNIPPET 20.11
def _pickle_method(method):
    func_name=method.__func__.__name__
    obj=method.__self__
    cls=type(method.__self__)
    return _unpickle_method,(func_name,obj,cls)
#———————————————————————————————————————
def _unpickle_method(func_name,obj,cls):
    for cls in cls.mro():
        try:func=cls.__dict__[func_name]
        except KeyError:pass
        else:break
    return func.__get__(obj,cls)
#———————————————————————————————————————
import copyreg,types,multiprocessing as mp
copyreg.pickle(types.MethodType,_pickle_method,_unpickle_method)
# SNIPPET 20.12
def processJobsRedux(jobs,task=None,numThreads=24,redux=None,reduxArgs={},
                     reduxInPlace=False):
    '''
    Run in parallel
    jobs must contain a ’func’ callback, for expandCall
    redux prevents wasting memory by reducing output on the fly
    '''
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    imap,out,time0=pool.imap_unordered(expandCall,jobs),None,time.time()
    # Process asynchronous output, report progress
    for i,out_ in enumerate(imap,1):
        if out is None:
            if redux is None:out,redux,reduxInPlace=[out_],list.append,True
            else:out=copy.deepcopy(out_)
        else:
            if reduxInPlace:redux(out,out_,**reduxArgs)
            else:out=redux(out,out_,**reduxArgs)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    if isinstance(out,(pd.Series,pd.DataFrame)):out=out.sort_index()
    return out
# SNIPPET 20.13
def mpJobList(func,argList,numThreads=24,mpBatches=1,linMols=True,redux=None,
              reduxArgs={},reduxInPlace=False,**kargs):
    if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    jobs=[]
    for i in range(1,len(parts)):
        job={argList[0]:argList[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    out=processJobsRedux(jobs,redux=redux,reduxArgs=reduxArgs,
                         reduxInPlace=reduxInPlace,numThreads=numThreads)
    return out
# SNIPPET 20.14
# pcs=mpJobList(getPCs,('molecules',fileNames),numThreads=24,mpBatches=1,
#     path=path,eVec=eVec,redux=pd.DataFrame.add)
#———————————————————————————————————————
def getPCs(path,molecules,eVec):
    # get principal components by loading one file at a time
    pcs=None
    for i in molecules:
        df0=pd.read_csv(path+i,index_col=0,parse_dates=True)
        if pcs is None:pcs=np.dot(df0.values,eVec.loc[df0.columns].values)
        else:pcs+=np.dot(df0.values,eVec.loc[df0.columns].values)
    pcs=pd.DataFrame(pcs,index=df0.index,columns=eVec.columns)
    return pcs
