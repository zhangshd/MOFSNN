import sys
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors

# This is called by example_feature_generation.py.

def main():

	# user command line inputs
	structure_path = sys.argv[1]
	name = sys.argv[2] # name of the MOF
	RACs_folder = sys.argv[3]
	wiggle_room = float(sys.argv[4])

	# result log
	f = open(f'{RACs_folder}/RAC_getter_log.txt', 'w')

	try:
		# Makes and populates the linkers and sbus folders.
		full_names, full_descriptors = get_MOF_descriptors(f'{structure_path}', 3, path = RACs_folder, xyzpath = f'{RACs_folder}/{name}.xyz', wiggle_room=wiggle_room, 
			max_num_atoms=6000, get_sbu_linker_bond_info=True, surrounded_sbu_file_generation=True, detect_1D_rod_sbu=True); # Allowing for very large unit cells.

		if (len(full_names) <= 1) and (len(full_descriptors) <= 1): # This is a featurization check
			f.write('FAILED - Featurization error')
			f.close()
	except ValueError:
		f.write('FAILED - ValueError')
		f.close()
	except NotImplementedError:
		f.write('FAILED - NotImplementedError')
		f.close()
	except AssertionError:
		f.write('FAILED - AssertionError')
		f.close()


if __name__ == "__main__":
	main()
