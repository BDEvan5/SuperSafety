import glob , yaml, csv


class DataBuilder:
    def __init__(self, eval_name=None):
        self.data = {}
        self.base_keys = []
        self.n = 0
        self.path = f"Data/Vehicles/"
        self.eval_name = eval_name

        self.build_keys()
        self.read_data()
        if self.eval_name is None:
            self.save_data_table()
            self.save_eval("all")
        else:
            self.save_eval(eval_name)

    def build_keys(self):
        with open(f"Data/base_key_builder.yaml") as f:
            key_data = yaml.safe_load(f)

        for key in key_data:
            self.base_keys.append(key)

    def read_data(self):
        folders = glob.glob(f"{self.path}*/")
        for i, folder in enumerate(folders):
            print(f"Folder being opened: {folder}")
            
            try:
                config = glob.glob(folder + '/*_record.yaml')[0]
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Filename issue: {folder}")
                continue            

            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                continue

            if config_data['eval_name'] != self.eval_name and self.eval_name is not None:
                continue

            self.data[i] = {}
            for key in config_data.keys():
                if key == "SSS" or key == "Wo":
                    for sub_key in config_data[key].keys():
                        store_key = f"{key}_{sub_key}"
                        self.data[i][store_key] = config_data[key][sub_key]
                        if not store_key in self.base_keys:
                            self.base_keys.append(store_key)
                    continue
                if key in self.base_keys:
                    self.data[i][key] = config_data[key]

    def save_data_table(self, name="DataTable"):
        directory = "DataAnalysis/" + name + ".csv"
        with open(directory, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=self.base_keys)
            writer.writeheader()
            for key in self.data.keys():
            # for i in range(len(self.data.keys())):
                writer.writerow(self.data[key])


        print(f"Data saved to {name} --> {len(self.data)} Entries")

    def save_eval(self, eval_name):
        directory = "Data/Results/Data_" + eval_name + ".csv"
        with open(directory, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=self.base_keys)
            writer.writeheader()
            for key in self.data.keys():
                writer.writerow(self.data[key])


        print(f"Data saved to {eval_name} --> {len(self.data)} Entries")

#TODO: in the future, read it in once and then just save it specifically according to eval

def run_builder():
    # DataBuilder()
    # DataBuilder('KernelGen')
    # DataBuilder('benchmark')
    # DataBuilder('repeatability')
    DataBuilder('RewardTest')


if __name__ == "__main__":
    run_builder()


