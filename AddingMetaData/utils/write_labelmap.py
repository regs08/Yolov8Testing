import os

label_file_dir = "/label_files"
label_file_name = 'coco_labels.txt'
label_file_path = os.path.join(label_file_dir, label_file_name)

def write_label_map(label_file:str, outdir:str):
    if os.path.exists(label_file) and os.path.exists(outdir):
        out_file = f"{os.path.basename(label_file)[:-4]}_label_map.txt"
        out_path = os.path.join(outdir, out_file)
        with open(label_file_path, 'w') as file:
            file.write("{\n")
            for index, label in enumerate(label_file):
                file.write(f"    {index}: '{label}',\n")
            file.write("}\n")
