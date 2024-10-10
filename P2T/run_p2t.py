import os

# all filenames
padding_files = [
('polynomialmul_polyhedral_model.txt', 'polynomialmul_array_size.txt'),
('IMA_polyhedral_model.txt', 'IMA_array_size.txt'),
('NAT_polyhedral_model.txt', 'NAT_array_size.txt'),
('SOAI_polyhedral_model.txt', 'SOAI_array_size.txt')
]

shifting_files = [
('star5_polyhedral_model.txt', 'star5_array_size.txt'),
('box25_polyhedral_model.txt', 'box25_array_size.txt')
]

none_files = [
('2mm_polyhedral_model.txt', '2mm_array_size.txt'),
('3mm_polyhedral_model.txt', '3mm_array_size.txt'),
('atax_polyhedral_model.txt', 'atax_array_size.txt'),
('bicg_polyhedral_model.txt', 'bicg_array_size.txt'),
('mvt_polyhedral_model.txt', 'mvt_array_size.txt'),
('trmm_polyhedral_model.txt', 'trmm_array_size.txt')
]

optimization_methods = {
'padding': padding_files,
'shifting': shifting_files,
'none': none_files
}

# run p2t.py
def run_p2t(polyhedral_model, array_size, method):
    command = f"python3 P2T.py {polyhedral_model} {array_size} {method}"
    print(f"Running: {command}")
    os.system(command)

for method, files in optimization_methods.items():
    for poly_model, array_size in files:
        run_p2t(poly_model, array_size, method)