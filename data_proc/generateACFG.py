'''
    This material is based upon work supported by the
    Defense Advanced Research Projects Agency (DARPA)
    and Naval Information Warfare Center Pacific
    (NIWC Pacific) under Contract Number N66001-20-C-4024.

    The views, opinions, and/or findings expressed are
    those of the author(s) and should not be interpreted
    as representing the official views or policies of
    the Department of Defense or the U.S. Government.
'''

import subprocess
import os
import shutil
import argparse
from tqdm import tqdm


ghidra_vars = ['GHIDRA_HEADLESS', 'GHIDRA_SCRIPTS', 'GHIDRA_PROJECT']

def init_ghidra_paths():
	try:
		ghidra_dir = r"/home/kyle/ghidra_10.0_PUBLIC_20210621"
		#ghidra_scripts = os.environ['GHIDRA_SCRIPTS']
		mindsight = r"/home/kyle/TreeEmbedding/data_proc"

		tmpGhidraProj = os.getcwd() + '/tmpProj'
		ghidra_project = r"/home/kyle/treeEmbed"

		global GHIDRA_HEADLESS
		global GHIDRA_SCRIPTS
		global GHIDRA_PROJECT

		GHIDRA_HEADLESS = os.path.join(ghidra_dir, 'support')
		GHIDRA_SCRIPTS = os.path.join(mindsight, 'ghidra_scripts')
		GHIDRA_PROJECT = ghidra_project
		
	except KeyError:
		print("Could not find environment variables: GHIDRA_INSTALL_DIR and MINDSIGHT")
		#print("Could not find environment variables: GHIDRA_INSTALL_DIR and GHIDRA_SCRIPTS")

def start_acfg_errorlog(args):
    with open("acfgErrLog.txt", "w") as file:
        file.write("Error Log for generateACFG.py")
        file.write("\nArguments generated:")
        file.write("\n\tDirectory: %s "%args.directory)
        file.write("Create Stripped: %s Overwrite: %s"%(args.create_stripped, args.overwrite))
        file.write("\n\tGHIDRA_HEADLESS: %s"%GHIDRA_HEADLESS)
        file.write("\n\tGHIDRA_SCRIPTS: %s"%GHIDRA_SCRIPTS)
        file.write("\n\tGHIDRA_PROJECT: %s"%GHIDRA_PROJECT)

def validate_database(db_directory):
	CHECK_THRES = 20
	valid_pkgs = 0

	pkg_paths = [ p.path for p in os.scandir(db_directory) if p.is_dir() ]

	for pkg in pkg_paths[:CHECK_THRES]:
		# go to pkg_arch/bin and get all binaries in folder
		bin_path = os.path.join(pkg, "bin")
		pkg_bins = [f for f in os.listdir(bin_path) if os.path.isfile(os.path.join(bin_path, f))]

		if os.path.isdir(bin_path) and len(pkg_bins):
			valid_pkgs += 1

	if(valid_pkgs < CHECK_THRES/2):
		raise ValueError("Provided invalid directory. Directory does not contain expected structure.")

def sbin_exist(binary, dst_path):
	base = os.path.splitext(binary)[0]
	if os.path.isfile("{}.sbin".format(base)):
		return True
	return False

def generate_sbin(binary, bin_path, out_path):
	bin_name = os.path.abspath(bin_path+'/'+binary)

	sbin = os.path.splitext(binary)[0] + ".sbin"
	sbin_name = os.path.abspath(out_path+'/'+sbin) 

	cmd = 'strip -s {} -o {}'.format(bin_name, sbin_name)
	rc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	if rc.returncode:
		with open("acfgErrLog.txt", "a") as file:
			file.write("\nError from `strip`:")
			file.write(rc.stderr.decode("utf-8"))
	
	return rc.returncode

def generate_db_sbin(db_directory, overwrite=False):
	errors = 0
	skipped = 0

	# get all package folders in database
	pkg_paths = [ p.path for p in os.scandir(db_directory) if p.is_dir() ]

	sbinBar = tqdm(total = len(pkg_paths), desc="Generating stripped binaries...")
	for pkg in pkg_paths:
		# go to pkg_arch/bin and get all binaries in folder
		bin_path = os.path.join(pkg, "bin")
		pkg_bins = [f for f in os.listdir(bin_path) if os.path.isfile(os.path.join(bin_path, f))]

		for binary in pkg_bins:
			output_exists = sbin_exist(binary, bin_path)
			if not output_exists or overwrite:
				errors += generate_sbin(binary, bin_path, bin_path)
			else:
				skipped += 1

		sbinBar.update(1)

	sbinBar.close()
	return (errors, skipped)


def binary_acfg_exist(binary, dst_path):
	gout = os.path.abspath('/tmp/{}-acfg'.format(binary))
	if os.path.isdir(gout):
		return True

	dst_out = os.path.join(dst_path,'{}-acfg'.format(binary))
	if os.path.isdir(dst_out):
		return True

	return False

def generate_bin_acfg(binary, bin_path, out_path):

	if not all(var in globals() for var in ghidra_vars):
		init_ghidra_paths()
	
	binary_name = os.path.join(bin_path, binary)
	# Split GHIDRA_PROJECT into project path and project name
	project_path = os.path.dirname(GHIDRA_PROJECT)
	project_name = os.path.basename(GHIDRA_PROJECT)

	# run ASTGenerator.java
	# Read the docs: https://ghidra.re/ghidra_docs/analyzeHeadlessREADME.html
	cmd = '{}/analyzeHeadless {} {} \
		-scriptPath {} \
		-import {} \
		-postScript ASTGenerator.java \
		-readOnly'.format(GHIDRA_HEADLESS, project_path, project_name, GHIDRA_SCRIPTS, binary_name)

	# TODO: Look into details in how/why errors occur
	rc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	# move generated /tmp/binary-acfg/ and /tmp/binary-ccode/ to pkg_arch/acfg/
	acfg_out = os.path.abspath('/tmp/{}-acfg'.format(binary))
	ccode_out = os.path.abspath('/tmp/{}-CCode'.format(binary))

	if os.path.isdir(acfg_out) and (rc.returncode >= 0):
		shutil.move(acfg_out, out_path)

		if os.path.isdir(ccode_out):
			shutil.move(ccode_out, out_path)
		return 0

	else:
		with open("acfgErrLog.txt", "a") as file:
			file.write("\nError: %s"%binary_name)
			file.write(rc.stderr.decode("utf-8"))
			file.write(rc.stdout.decode("utf-8"))

		return -1

# bin_path = "/home/kali/workspace/eecs299/ALLSTAR/FetchFiles/test_amd64/libpangomm-1.4-dev_amd64/bin/"
# binary = "libpangomm-1.4.so.1.0.30.bin"

def generate_db_acfg(db_directory, overwrite=False):
	created = 0
	skipped = 0
	errors = 0

	# Get all package folders in database
	pkg_paths = [ p.path for p in os.scandir(db_directory) if p.is_dir() ]

	dbBar = tqdm(total = len(pkg_paths), desc="Generating ACFG...")
	for pkg in pkg_paths:

		# Get all binary files in folder pkg_arch/bin/
		bin_path = os.path.join(pkg, "bin")
		pkg_bins = [f for f in os.listdir(bin_path) if os.path.isfile(os.path.join(bin_path, f))]

		# If binaries exist create pkg_arch/acfg/ if not already present
		acfg_path = os.path.join(pkg, "acfg")
		if not os.path.isdir(acfg_path) and len(pkg_bins):
			os.mkdir(acfg_path)

		for binary in pkg_bins:
			dbBar.set_description("Generating ACFG for %s/bin/%s..."%(os.path.basename(pkg),binary))

			output_exists = binary_acfg_exist(binary, acfg_path)
			if (not output_exists) or overwrite:
				if(generate_bin_acfg(binary, bin_path, acfg_path) >= 0):
					created += 1
				else:
					errors += 1
			else:
				skipped += 1

		dbBar.update(1)
	dbBar.close()

	return (created, skipped, errors)


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser()

	group = parser.add_argument_group('REQUIRED ARGS')
	group.add_argument('-d','--directory',default='./',type=str,help='Database directory and output parent directory')

	group2 = parser.add_argument_group('OPTIONAL ARGS')
	group2.add_argument('--create-stripped',action='store_true',help='Create stripped copies of unstripped binaries before generating ACFGs')
	group2.add_argument('-w','--overwrite',action='store_true',help='Overwrite ACFG files, including any other files generated within script (CCcode and stripped binaries)')

	args = parser.parse_args()

	try:
		db_dir = args.directory
		db_dir = os.path.abspath(db_dir)
		if not os.path.isdir(db_dir):
			raise ValueError("Passed directory invalid")

		validate_database(db_dir)

	except:
		print("\nErr: Entered invalid directory as an argument\n")
		quit()

	init_ghidra_paths()
	start_acfg_errorlog(args)
	
	if(args.create_stripped):
		(sbin_errs, sbin_skip) = generate_db_sbin(db_dir, args.overwrite)

	(created, skipped, errors) = generate_db_acfg(db_dir, args.overwrite)

	print("\nGenerated ACFG files for database %s:"%db_dir)
	if(args.create_stripped): print("Included stripped binaries: Errors: %d Skipped: %d"%(sbin_errs, sbin_skip))
	print("ACFGs Created: %d  Binaries Skipped: %d  Errors: %d"%(created, skipped, errors))






