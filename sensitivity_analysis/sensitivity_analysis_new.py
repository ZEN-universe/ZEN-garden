import os
import yaml
import random
import logging
import shutil
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
from eth_colors import ETHColors

from zen_garden._internal import main
from zen_garden.postprocess.results import Results

class SensitivityAnalysis():
    file_format = ".csv"

    def __init__(self, config="../data/config.py", suffix_ov_file="_origin", dataset=None):
        """init elementary effect"""
        self.set_config(config)
        self.set_dataset(dataset)
        self.set_results_path()
        self.set_figures_path()
        self.set_eth_colors()
        self.update_config()

        self.set_dict_params()
        self.set_dict_vars()
        self.set_suffix_ov_files(suffix_ov_file)
        self.set_file_name_attributes()
        self.set_suffix_attributes()
        # self.restore_reference_scenario()
        # self.save_original_input_files()


    def set_results_path(self):
        """set results path"""
        self.results_path = os.path.join("..","sensitivity_analysis","outputs_"+self.dataset)
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

    def set_figures_path(self):
        """set results path"""
        self.figures_path  = os.path.join(self.results_path,  "figures_"+self.dataset)
        if not os.path.exists(self.figures_path):
            os.mkdir(self.figures_path)

    def set_eth_colors(self):
        """set ETH colors"""
        self.eth_colors = ETHColors()

    def set_results_folder(self, morris_method=True , name = None):
        """set results folder and check if folder exists"""
        if name:
            self.results_folder = name
        elif morris_method:
            self.results_folder = "morris_screening_method"
        else:
            self.results_folder = "local_sensitivity_analysis"
        # create results folders
        path = os.path.join(self.results_path, self.results_folder)
        if not os.path.exists(path):
            os.mkdir(path)

    def set_config(self, config):
        """ import config file to run optimization
        :param config: folderpath to config file
        :return config: dictionary with configs"""
        config_path, config_file = os.path.split(config)
        self.set_folder_path(config_path)
        os.chdir(config_path)
        spec = importlib.util.spec_from_file_location("module", config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.config = module.config


    def set_dataset(self, dataset):
        """set dataset path"""
        if not dataset:
            self.dataset_path = self.config.analysis["dataset"]
            self.dataset = os.path.split(self.dataset_path)[-1]
        else:
            self.dataset = dataset
            self.dataset_path = os.path.join(self.folder_path, dataset)
            self.config.analysis["dataset"] = self.dataset_path


    def set_folder_path(self,folder_path):
        """set folder path"""
        self.folder_path = folder_path

    def set_n_runs(self,n_runs):
        """set number of runs"""
        self.n_runs = n_runs

    def set_n_levels(self,n_levels):
        """set number of levels following Satlli's "Global Sensitivity Analysis. The Primer" (2007)"""
        self.n_levels = n_levels

    def set_dict_params(self, file_name="uncertain_parameters.yml"):
        """get uncertain input parameter names and set number of uncertain input parameters
        :param file_name: name of yml file containing information about uncertain input parameters"""
        path = os.path.join(self.dataset,file_name)
        with open(path) as f:
            input_parameters = yaml.safe_load(f)
        # create param dict
        self.param_dict = dict()
        for set, element_dict in input_parameters.items():
            for element, params in element_dict.items():
                if element == "None":
                    elements = self.config.system[set]
                else:
                    elements = [element]
                for element in elements:
                    # iterate through elements and params
                    for param, values in params.items():
                        self.param_dict[f"{param}_{element}"]={#"path": os.path.join(self.dataset_path,set,element),
                                                               "element": element,
                                                               "param": param, **values}
        self.n_parameters = len(self.param_dict.keys())

    def set_dict_vars(self, file_name="output_variables.yml"):
        """get output variables of interest
        :param file_name: name of yml file containing information about output variables of interest
        :return output_dict: dictionary with information about output variables of interest"""
        path = os.path.join(self.dataset,file_name)
        with open(path) as f:
            input_parameters = yaml.safe_load(f)
        # create param dict
        self.var_dict = dict()
        for set, element_dict in input_parameters.items():
            if set == "None":
                _dict = {var: {"var": var, "element": None} for var in element_dict}
                self.var_dict.update(_dict)
            else:
                for element, var_list in element_dict.items():
                    if element == "None":
                        # iterate through elements
                        elements = self.config.system[set]
                        _dict = {f"{var}_{element}": {"var": var, "element": element} for element in elements for var in var_list}
                    else:
                        _dict = {f"{var}_{element}": {"var": var, "element": element} for var in var_list}
                    self.var_dict.update(_dict)
        # add LCOH
        self.var_dict.update({"LCOH": {"var": "net_present_cost", "element": "compute"}})

    def set_n_trajectories(self):
        """set number of trajectories"""
        self.n_trajectories = int(max(1, np.floor(self.n_runs / (self.n_parameters + 1))))

    def set_trajectory_params(self):
        """set up empty trajectory array"""
        self.trajectory_step_size = self.n_levels / (2 * self.n_levels - 1)
        self.uncertain_params = list(self.param_dict.keys())
        self.samples = np.arange(0,(self.n_parameters + 1))
        # create trajectory xarray
        coords = {"samples": self.samples, "uncertain_params": self.uncertain_params}
        trajectory = np.full((len(self.samples), self.n_parameters), np.nan)
        self.xr_trajectory = xr.DataArray(trajectory, coords=coords, dims=coords.keys())

    def set_file_name_attributes(self, name="attributes"):
        """set attributes name"""
        self.attributes_name = name

    def set_suffix_attributes(self, suffix="_default"):
        """set suffix for attribute params"""
        self.suffix_attribute_files = suffix

    def set_suffix_ov_files(self,suffix):
        """set appendix to save original files"""
        self.suffix_ov_file = suffix

    def get_xarray_results(self, morris_method: bool):
        """create xarray to save results"""
        vars = list(self.var_dict.keys())
        if morris_method:
            trajectories = np.arange(0, self.n_trajectories)
            coords = {"trajectory": trajectories, "sample": self.samples, "var": vars}
            results = np.full((len(trajectories), len(self.samples), len(vars)), np.nan)
        else:
            params = list(self.param_dict.keys())
            coords = {"param": params, "sample": self.samples, "var": vars}
            results = np.full((len(params), len(self.samples), len(vars)), np.nan)
        results = xr.DataArray(data=results, dims=list(coords.keys()), coords=coords)
        return results

    def get_xarray_elementary_effects(self):
        """create results dict to save results for the different trajectories"""
        params = list(self.param_dict.keys())
        vars = list(self.var_dict.keys())
        trajectories = np.arange(0, self.n_trajectories)
        coords = {"trajectory": trajectories, "param": params, "var": vars}
        results = np.full((len(trajectories), len(params), len(vars)), np.nan)
        elementary_effects = xr.DataArray(data=results, dims=list(coords.keys()), coords=coords)
        return elementary_effects

    def get_xarray_trajectory(self):
        """return empty trajectory"""
        return self.xr_trajectory.copy(deep=True)

    def get_trajectory(self):
        """determine trajectories depending on number of runs and number of parameters
        :return trajectory: return normalized trajectory"""
        # get empty trajectory
        trajectory = self.get_xarray_trajectory()
        # random starting point
        trajectory[self.samples[0], :] = (np.random.randint(1, self.n_levels, size=self.n_parameters) - 1) / (self.n_levels - 1)
        # random parameter sequence
        parameter_sequence = random.sample(self.uncertain_params, self.n_parameters)
        # determine parameter sequences
        for sample in self.samples[1:]:
            previous_level = trajectory.loc[sample-1, parameter_sequence[sample-1]]
            # if reduction out of range > increase
            if previous_level - self.trajectory_step_size < -0.1 * self.trajectory_step_size:
                new_level = previous_level + self.trajectory_step_size
            # if increase out of range > decrease
            elif previous_level + self.trajectory_step_size > 1 + 0.1 * self.trajectory_step_size:
                new_level = previous_level + self.trajectory_step_size
            # else random step direction with equal up/down probability
            else:
                new_level = previous_level + (1 - 2 * (np.random.random() >= 0.5)) * self.trajectory_step_size
            trajectory.loc[sample, :] = trajectory.loc[sample-1, :]
            trajectory.loc[sample, parameter_sequence[sample-1]] = new_level
        return trajectory

    def get_results_name(self, t, s):
        """get results name for sample s of given trajectory or parameter
        :param t: trajectory or parameter
        :param s: sample s of trajectory or parameter
        :return name: results name"""
        if t in self.param_dict.keys():
            name = f"parameter_{t}_sample_{s}"
        elif isinstance(t, int):
            name = f"trajectory_{t}_sample_{s}"
        else:
            name = f"{t}_sample_{s}"
        return name

    def update_config(self):
        """update config and load system.py and scenario.py"""
        # update system
        system_path = os.path.join(self.dataset_path, "system.py")
        if not os.path.exists(system_path):
            raise FileNotFoundError(f"system.py not found in dataset: {self.dataset_path}")
        spec = importlib.util.spec_from_file_location("module", system_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.config.system.update(module.system)
        # update scenario
        system_path = os.path.join(self.dataset_path, "scenarios.py")
        if not os.path.exists(system_path):
            raise FileNotFoundError(f"scenarios.py not found in dataset: {self.dataset_path}")
        spec = importlib.util.spec_from_file_location("module", system_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.config.scenarios.update(module.scenarios)

    def save_original_input_files(self):
        """save original input data files"""
        for param, values in self.param_dict.items():
            path = values["path"]
            param = values["param"]
            # attributes.csv
            if not os.path.exists(os.path.join(path, self.attributes_name + self.suffix_ov_file + self.file_format)):
                attributes = pd.read_csv(os.path.join(path, self.attributes_name + self.file_format))
                attributes.to_csv(os.path.join(path, self.attributes_name + self.suffix_ov_file + self.file_format),
                                  index=False)
            # param.csv
            if os.path.exists(os.path.join(path, param + self.file_format)) and \
                    not os.path.exists(os.path.join(path, param + self.suffix_ov_file + self.file_format)):
                df_input = pd.read_csv(os.path.join(path, param + self.file_format))
                df_input.to_csv(os.path.join(path, param + self.suffix_ov_file + self.file_format), index=False)

    def update_input_data(self, sample):
        """update input data files to current sample"""
        for uncertain_param in sample.coords["uncertain_params"].data:
            assert uncertain_param in self.param_dict.keys(), f"Parameter {uncertain_param} is not defined as an uncertain parameter"
            values = self.param_dict[uncertain_param]
            path = values["path"]
            param = values["param"]
            # load attributes
            attribute_param = param +self.suffix_attribute_files
            assert os.path.exists(os.path.join(path, self.attributes_name+self.suffix_ov_file+self.file_format)), \
                f"original attribute file for {uncertain_param} is missing"
            attributes = pd.read_csv(os.path.join(path, self.attributes_name + self.suffix_ov_file + self.file_format), index_col=0)
            # check units #TODO implement a more sophisticated way to check and update units
            if attributes.loc[attribute_param,"value"] == 0:
                logging.warning(f"Attribute value of {param} is set to 1 such that lb and ub can be computed")
                attributes.loc[attribute_param, "value"] = 1
            if param != "conversion_factor":
                lb = float(attributes.loc[attribute_param, "value"]) * values["lb"]/values["ref"]
                ub = float(attributes.loc[attribute_param, "value"]) * values["ub"]/values["ref"]
                attributes.loc[attribute_param, "value"] = lb + float(sample.loc[uncertain_param]) * (ub - lb)
                if "lifetime" in param:
                    attributes.loc[attribute_param, "value"] = round(attributes.loc[attribute_param, "value"],0)
            attributes.to_csv(os.path.join(path, self.attributes_name + self.file_format), index=True)
            # check if a file with param values exists
            # TODO implement method how to deal with distance dependent params (if distance is a parameter)
            if os.path.exists(os.path.join(path,param+self.suffix_ov_file+self.file_format)):
                df_input = pd.read_csv(os.path.join(path,param+self.suffix_ov_file+self.file_format))
                if param == "conversion_factor":
                    df_input = df_input.set_index(df_input.columns[0])
                    for c, carrier in enumerate(df_input.columns):
                        if carrier in values.keys():
                            vals = values[carrier]
                            lb = df_input.iloc[:-1,c].astype(float) * vals["lb"] / vals["ref"]
                            ub = df_input.iloc[:-1,c].astype(float) * vals["ub"] / vals["ref"]
                            df_input.iloc[:-1,c] = lb + sample.loc[uncertain_param].values * (ub - lb)
                else:
                    df_input = df_input.set_index(list(df_input.columns[:-1]))
                    lb = df_input * values["lb"]/values["ref"]
                    ub = df_input * values["ub"]/values["ref"]
                    df_input = lb + sample.loc[uncertain_param].values * (ub - lb)
                df_input.to_csv(os.path.join(path, param + self.file_format), index=True)

    def compute_elementary_effects(self, t, trajectory):
        """calculate elementary effects of trajectory t
        :parameters t: trajectory t
        :param trajectory: trajectory xarray"""
        for s in self.samples[1:]:
            d_input = np.abs(trajectory[s - 1, :] - trajectory[s])
            assert len(d_input.where(d_input > 0, drop=True)) == 1, "more than one param value changed"
            param = d_input.where(d_input > 0, drop=True).get_index(d_input.dims[0])[0]
            d_output = np.abs(self.results.loc[t, s - 1, :] - self.results.loc[t, s, :])
            self.elementary_effects.loc[t, param, :] = d_output / d_input.loc[param]

    def calculate_elementary_effects_statistics(self):
        """calculate elementary effect statistics"""
        # xarray for normalized results
        elementary_effects = self.elementary_effects.copy(deep=True)
        # xarray for elementary effect statistics
        params = list(self.param_dict.keys())
        vars = list(self.var_dict.keys())
        statistics = ["mean", "std_dev", "abs_mean"]
        coords = {"statistic": statistics, "param": params, "var": vars}
        ee_statistics = np.full((len(statistics), len(params), len(vars)), np.nan)
        self.ee_statistics = xr.DataArray(data=ee_statistics, dims=list(coords.keys()), coords=coords)
        for var in vars:
            for param in params:
                # normalize elementary effects
                norm = np.linalg.norm(self.elementary_effects.loc[:, param, var]) #alternatively, normalize with std.dev?
                if norm != 0:
                    elementary_effects.loc[:, param, var] /= norm
                else:
                    elementary_effects.loc[..., var] *= 0
                # compute mean, std_dev and absolute mean
                self.ee_statistics.loc["mean", param, var] = elementary_effects.loc[:, param, var].mean()
                self.ee_statistics.loc["std_dev", param, var] = elementary_effects.loc[:, param, var].std()
                self.ee_statistics.loc["abs_mean", param, var] = np.mean(np.abs(elementary_effects.loc[:, param, var]))

    def save_results(self, morris_method: bool):
        """save results of sensitivity analysis"""
        path = os.path.join(self.results_path, self.results_folder)
        self.results.to_dataframe(name="value").to_csv(os.path.join(path, "results.csv"))
        if morris_method:
            self.elementary_effects.to_dataframe(name="value").to_csv(os.path.join(path, "elementary_effects.csv"))
            self.ee_statistics.to_dataframe(name="value").to_csv(os.path.join(path, "elementary_effects_statistics.csv"))

    def morris_screening_method(self, n_runs = 500, n_levels=4):
        """perform morris screening method"""
        ## TODO implement with new scenarios
        self.set_n_runs(n_runs)
        self.set_n_levels(n_levels)
        self.set_results_folder(morris_method=True)
        self.set_n_trajectories()
        self.set_trajectory_params()
        self.results = self.get_xarray_results(morris_method=True)
        self.elementary_effects = self.get_xarray_elementary_effects()
        scenarios = dict()
        for t in range(self.n_trajectories):
            trajectory = self.get_trajectory()
            # create scenario dict
            scenario = {f"trajectory_{t}_sample_{s}": {
                            # element
                            self.param_dict[name]["element"]: {
                                # param and default op
                                self.param_dict[name]["param"]: {"default_op": float(trajectory.loc[s,name])}}
                            for name in trajectory.coords["uncertain_params"].values}
                            for s in self.samples}
            scenarios.update(scenario)
        return scenarios
        #
        # self.compute_elementary_effects(t, trajectory)
        # self.calculate_elementary_effects_statistics()
        # # save results
        # self.save_results(morris_method=True)
        # self.elementary_effects.to_dataframe().to_csv()

    def restore_reference_scenario(self):
        """restore reference scenario"""
        for param, values in self.param_dict.items():
            path = values["path"]
            param = values["param"]
            # overwrite attributes
            old_file = os.path.join(path, self.attributes_name + self.suffix_ov_file + self.file_format)
            if os.path.exists(old_file):
                new_file = os.path.join(path, self.attributes_name + self.file_format)
                shutil.copy(old_file, new_file)
            # overwrite param file
            old_file = os.path.join(path,param+self.suffix_ov_file+self.file_format)
            if os.path.exists(old_file):
                new_file = os.path.join(path, param + self.file_format)
                shutil.copy(old_file, new_file)

    def local_sensitivity_analysis(self, n_levels=None):
        """perform local sensitivity analysis"""
        if n_levels:
            self.set_n_levels(n_levels)
        self.set_results_folder(morris_method=False)
        step_size = 1/self.n_levels
        self.samples = np.round(np.arange(0,1+step_size,step_size),2)
        self.results = self.get_xarray_results(morris_method=False)
        count=0
        for param in self.param_dict.keys():
            count +=1
            for s in self.samples:
                sample = self.get_sample(uncertain_params=param,samples=s)
                self.update_input_data(sample)
                self.run_zen_garden(param,s)
            # if count >2:
            #     break
        self.save_results(morris_method=False)

    def get_sample(self, uncertain_params, samples, data=None):
        """get sample xarray to overwirte input params"""
        if not data:
            data = samples
        if not isinstance(uncertain_params, list):
            uncertain_params = [uncertain_params]
        if not isinstance(samples, list):
            samples = [samples]
        coords = {"uncertain_params": uncertain_params, "samples": samples}
        sample = xr.DataArray(data=data, dims=list(coords.keys()), coords=coords)
        return sample

    def get_job_index(self, scenario):
        """get job index for given scenario"""
        if not hasattr(self,"job_index_dict"):
            count = 0
            self.job_index_dict = dict()
            for scen, scen_dict in self.config.scenarios.items():
                no_runs = 1
                for element, element_dict in scen_dict.items():
                    if element not in ["system", "analysis"]:
                        for param, param_dict in element_dict.items():
                            if "default_op" in param_dict.keys() and isinstance(param_dict["default_op"], list):
                                no_runs *= len(param_dict["default_op"])
                self.job_index_dict[scen] = np.arange(count,count+no_runs,1)
                count += no_runs
        return self.job_index_dict[scen]

    def explore_2d_param_space(self, scenario):
        """explore a 2D parameter space"""
        if not scenario.startswith("scenario"):
            scenario="scenario_"+scenario
        # check for existing results
        if os.path.exists(os.path.join(self.results_path, scenario, scenario + ".csv")):
            return pd.read_csv(os.path.join(self.results_path,  scenario, scenario + ".csv"), index_col=[0])
        # create results folder and load results
        self.set_results_folder(name=scenario)
        res = Results(os.path.join("outputs", self.dataset, scenario))
        subscenarios = res.scenarios
        # extract results
        df_results = pd.DataFrame()
        # net present cost
        discount = 1 / (1 + 0.06)  # todo get discount rate form result dict
        techs = [tech for tech in ["SMR_CCS", "electrolysis"] if tech in res.results["system"]["set_conversion_technologies"]]
        hydrogen_production = res.get_total("flow_conversion_output").loc[subscenarios,techs,"hydrogen"].groupby(level=[0]).sum()
        present_hydrogen_production = sum(discount ** y * hydrogen_production[y] for y in hydrogen_production.columns)
        present_cost = res.get_total("net_present_cost").sum(axis=1)  # [M€]
        df_results["lcoh"] = present_cost/(present_hydrogen_production/33.3)  # [M€/kt]=[€/kg]
        # capacity addition
        capacity_addition = res.get_total("capacity_addition").groupby(level=[1,0]).sum()
        for tech in capacity_addition.index.unique("technology"):
            name = f"capacity_addition_{tech}"
            df_results[name] = capacity_addition.loc[tech].sum(axis=1)
        df_results.to_csv(os.path.join(self.results_path, self.results_folder, f"results_{scenario}.csv"), index=True)
        # parameter values
        elements = [element for element in res.results["scenarios"][subscenarios[0][9:]].keys()
                    if element not in ["base_scenario", "sub_folder","param_map"]]
        params = []
        for element in elements:
            param = list(res.results["scenarios"][res.scenarios[0][9:]][element].keys())
            assert len(param)==1, "More than one parameter is changed"
            param = param[0]
            if param == "capex_specific" and element in self.config.system["set_conversion_technologies"]:
                param = param + "_conversion"
            elif "set" in param:
                continue
            if param not in params:
                params.append(param)
        for param in params:
            param_vals = res.get_total(param)
            for level, values in enumerate(param_vals.index.levels):
                for element in elements:
                    if element in values:
                        tmp = param_vals.groupby(level=[level, 0]).mean()
                        df_results[f"{param}_{element}"] = tmp.loc[element].mean(axis=1)
                        continue
        # save results
        df_results.to_csv(os.path.join(self.results_path,scenario+".csv"), index=True)
        # delete results to reduce memory
        del res
        return df_results
    ### create plots

    def feasibility_map_3d(self, scenario: str):
        """feasibility map"""
        results = self.explore_2d_param_space(scenario)

        # get param values
        x_param, y_param = results.columns[-2:]
        x_support = results[x_param].unique()
        y_support = results[y_param].unique()

        # param values
        x = results[x_param].values
        y = results[y_param].values
        z = results["lcoh"].values
        # technologies
        if "capacity_addition_electrolysis" in results.columns:
            E = results["capacity_addition_electrolysis"].values
        else:
            E  = np.zeros(len(results.index))
        if "capacity_addition_SMR_CCS" in results.columns:
            SMR_CCS = results["capacity_addition_SMR_CCS"].values
        else:
            SMR_CCS = np.zeros(len(results.index))
        total = E+SMR_CCS
        # remove small values
        threshold = 0.011
        E[E/total<threshold] = 0
        SMR_CCS[SMR_CCS/total<threshold] = 0
        techs = np.empty(np.shape(E))
        techs[SMR_CCS>0] = 0
        techs[E>0] = 1
        techs[(E>0)==(SMR_CCS>0)] = 2
        # create meshgrid
        shape = (len(x_support), len(y_support))
        x = x.reshape(shape)
        y = y.reshape(shape)
        z = z.reshape(shape)
        techs = techs.reshape(shape)
        #markers = [self.get_marker(results, param_name, run) for run in runs]
        #ax.scatter(x, y, c=z, cmap='viridis',) # marker=markers
        # define colormap and number of levels
        cmap = {0: self.eth_colors.get_color("grey", 20),
                1: self.eth_colors.get_color("green", 80),
                2: self.eth_colors.get_color("red", 80)}
        colors = [cmap[key] for key in np.unique(techs)]
        n_levels = len(np.unique(techs)) - 1
        # create contour plot
        fig, ax = plt.subplots()
        ax.contourf(x, y, techs, colors=colors, alpha=0.5, levels=n_levels)  # plot technology selection with countourf
        #cmap = self.eth_colors.get_custom_colormaps(color="blue", skip_white=True)
        CS = ax.contour(x, y, z, 8, cmap=plt.colormaps.get_cmap("Dark2"))  # plot LCOH with contourlines
        labels=ax.clabel(CS, inline=True, colors="k", use_clabeltext=True)
        # manually define labelpositions
        # label_pos = []
        # for cline in CS.collections:
        #     for path in cline.get_paths():
        #         vertices = path.vertices
        #         vx = vertices[:, 0]
        #         vy = vertices[:, 1]
        #         xpos = np.min(vx) + (np.max(vx)-np.min(vx))/2
        #         ypos = np.min(vy) + (np.max(vy)-np.min(vy))/2
        #         label_pos.append((xpos, ypos))
        #
        # # remove old labels
        # for cline in CS.collections:
        #     cline.remove()
        # for label in labels:
        #     label.remove()
        # # add new contour lines and labels
        # CS_new = ax.contour(x, y, z, 6)
        # plt.clabel(CS_new, inline=True, manual=label_pos)
        #set axis
        ax.set_ylabel(y_param) #" ["+self.param_dict[y_param]["unit"]+"]"
        ax.set_xlabel(x_param) #" ["+self.param_dict[y_param]["unit"]+"]"
        # save figure
        plt.savefig(os.path.join(self.figures_path, scenario+".pdf"))
        plt.savefig(os.path.join(self.figures_path, scenario + ".png"))
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    # sa = SensitivityAnalysis()
    # run sensitivity analysis
    # sa.local_sensitivity_analysis(n_levels=4)

    sa = SensitivityAnalysis()
    sa.morris_screening_method(n_runs=500, n_levels=4)
    sa.feasibility_map_3d("B_price_import_electricity_natural_gas")
    sa.feasibility_map_3d("B_capex_electrolysis_price_import_electricity")
    sa.feasibility_map_3d("B_capex_electrolysis_price_import_natural_gas")
    sa.feasibility_map_3d("B_capex_electrolysis_SMR")
    sa.feasibility_map_3d("B_capex_electrolysis_CDR")
    sa.feasibility_map_3d("B_capex_electrolysis_CDR")
    sa.feasibility_map_3d("S_capex_SMR_price_import_natural_gas")

