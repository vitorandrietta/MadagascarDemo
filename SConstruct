import os, sys

try:
    import bldutil
    glob_build = True # scons command launched in RSFSRC
    srcroot = '../..' # cwd is RSFSRC/build/user/brcloud
    Import('env bindir libdir pkgdir')
    env = env.Clone()
except:
    glob_build = False # scons command launched in the local directory
    srcroot = os.environ.get('RSFSRC', '../..')
    sys.path.append(os.path.join(srcroot,'framework'))
    import bldutil
    env = bldutil.Debug() # Debugging flags for compilers
    bindir = libdir = pkgdir = None

targets = bldutil.UserSconsTargets()

# C mains
targets.c = '''
fdtd_brcloud
'''

# paradd
# anifd2d
# awefd

CC = env.get('CC')
if CC.rfind('icc') >= 0:
    env.Append(CCFLAGS=['-restrict','-wd188'])

targets.build_all(env, glob_build, srcroot, bindir, libdir, pkgdir)
