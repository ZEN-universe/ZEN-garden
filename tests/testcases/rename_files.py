import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    change_folder_name = {
    }
    change_file_name = {
        #Carrier
        "demand_carrier":"demand",
        "demand_carrier_yearly_variation": "demand_yearly_variation",
        "availability_carrier_import": "availability_import",
        "availability_carrier_export": "availability_export",
        "availability_carrier_import_yearly": "availability_import_yearly",
        "availability_carrier_export_yearly": "availability_export_yearly",
        "availability_carrier_import_yearly_variation": "availability_import_yearly_variation",
        "availability_carrier_export_yearly_variation": "availability_export_yearly_variation",

        "shed_demand_price_default": "price_shed_demand",
        "import_price_carrier": "price_import",
        "export_price_carrier": "price_export",
        "import_price_carrier_yearly_variation": "price_import_yearly_variation",
        "export_price_carrier_yearly_variation": "price_export_yearly_variation",

        #Technologies
        "linear_conver_factor": "conversion_factor",
        "existing_capacity":"capacity_existing",
        "existing_invested_capacity": "capacity_investment_existing",
        "max_built_capacity": "capacity_addition_max_",
        "min_built_capacity": "capacity_addition_min",
        "max_capacity_addition": "capacity_addition_max_",
        "min_capacity_addition": "capacity_addition_min",
        "loss_flow": "transport_loss_factor",
        "breakpoints_pwa_conver_efficiency": "breakpoints_pwa_conversion_factor",
        "nonlinear_conver_efficiency": "nonlinear_conversion_factor",
        "linear_conver_efficiency": "conversion_factor",

        "capex_per_distance": "capex_per_distance_transport",
        "opex_specific": "opex_specific_variable",
        "fixed_opex_specific": "opex_specific_fixed",

        #System
        "carbon_emissions_previous": "carbon_emissions_cumulative_existing",
        "carbon_price": "price_carbon_emissions",
        "carbon_price_overshoot": "price_carbon_emissions_overshoot",
    }
    # scenario files
    scenarios = {
        "demand_carrier": [3,4],
        "demand_carrier_yearly_variation": [5],
        "existing_capacity": [6],
    }
    for param, values in scenarios.items():
        tmp = {f"{param}_{val}": f"{change_file_name[param]}_{val}" for val in values}
        change_file_name.update(tmp)

    change_param_name={
        # Carrier
        "demand_carrier": "demand",
        "demand_carrier_yearly_variation": "demand_yearly_variation",
        "availability_carrier_import": "availability_import",
        "availability_carrier_export": "availability_export",
        "availability_carrier_import_yearly": "availability_import_yearly",
        "availability_carrier_export_yearly": "availability_export_yearly",
        "availability_carrier_import_yearly_variation": "availability_import_yearly_variation",
        "availability_carrier_export_yearly_variation": "availability_export_yearly_variation",

        "shed_demand_price": "price_shed_demand",
        "import_price_carrier": "price_import",
        "export_price_carrier": "price_export",
        "import_price_carrier_yearly_variation": "price_import_yearly_variation",
        "export_price_carrier_yearly_variation": "price_export_yearly_variation",

        # Technologies
        "linear_conver_efficiency": "conversion_factor",
        "existing_capacity": "capacity_existing",
        "existing_capacity_energy": "capacity_existing_energy",
        "existing_invested_capacity": "capacity_investment_existing",
        "existing_invested_capacity_energy": "capacity_investment_existing_energy",
        "max_built_capacity": "capacity_addition_max",
        "min_built_capacity": "capacity_addition_min",
        "max_built_capacity_energy": "capacity_addition_max_energy",
        "min_built_capacity_energy": "capacity_addition_min_energy",
        "loss_flow": "transport_loss_factor",

        "capex_per_distance": "capex_per_distance_transport",
        "opex_specific": "opex_specific_variable",
        "fixed_opex_specific": "opex_specific_fixed",
        "fixed_opex_specific_energy": "opex_specific_fixed_energy",

        # System
        "carbon_emissions_cumulative_existing": "carbon_emissions_limit",
        "carbon_price": "price_carbon_emissions",
        "carbon_price_overshoot": "price_carbon_emissions_overshoot",
        "previous_carbon_emissions": "carbon_emissions_cumulative_existing"
    }

    add_param_name = { #TODO only add param once and keep correct index
        # "set_conversion_technologies": ("capacity_addition_unbounded", np.inf),
        # "set_transport_technologies": ("capacity_addition_unbounded", np.inf),
        # "set_storage_technologies": ("capacity_addition_unbounded", np.inf),
        # "system_specification": ("market_share_unbounded_default", 0.1),
        # "system_specification": ("knowledge_spillover_rate_default", 0.025),
    }

    change_txt_name = {}

    file_type = ".csv"
    txt_type = ".txt"
    current_cwd = os.getcwd()
    datasets = ["test_1a", "test_1b", "test_1c",
                "test_2a", "test_2b", "test_2c",
                "test_3a", "test_3c",
                "test_4a", "test_4b", "test_4c", "test_4d", "test_4e", "test_4f", "test_4g",
                "test_5a",
                "test_6a",]
    for root,dirs,files in os.walk(current_cwd,topdown=False):
        if "pycache" in root or "output" in root:
            continue
        for folder in dirs:
            if folder in change_folder_name.keys():
                dir_old = os.path.join(root,folder)
                dir_new = os.path.join(root,change_folder_name[folder])
                os.rename(dir_old,dir_new)
        for file in files:
            if not file.endswith(".csv"):
                continue
            filepath = os.path.join(root, file)
            data = pd.read_csv(filepath, index_col=0)
            file_wo_suffix = file.replace(file_type,"")
            file_wo_txt_suffix = file.replace(txt_type,"")
            # update file contents
            if file_wo_suffix == "attributes":
                idx_dict = {idx: change_param_name[idx.replace("_default","")] + "_default" for idx in data.index
                            if idx.endswith("_default") and idx.replace("_default","") in change_param_name.keys()}
                if "technologies" in root:
                    _idx_dict_carrier = {idx: change_param_name[idx] for idx in data.index
                                        if "Carrier" in idx}
                    idx_dict.update(_idx_dict_carrier)
                # TODO test add data function
                # _add_attribute = {attr: val for set, (attr, val) in add_param_name.items() if set in root}
                # if _add_attribute:
                #     _add_data = pd.DataFrame.from_dict(_add_attribute,orient="index",columns=["value"])
                #     _add_data.index.name = data.index.name
                #     data = pd.concat([data, _add_data])
                data = data.rename(index=idx_dict)
                data.to_csv(filepath,index=True)
            else:
                col_dict =  {column: change_param_name[column] for column in data.columns
                             if column in change_param_name.keys()}
                data = data.rename(columns=col_dict)
                data.to_csv(filepath, index=True)
            # update filename
            if file_wo_suffix in change_file_name.keys():
                file_old = os.path.join(root,file)
                file_new = os.path.join(root,change_file_name[file_wo_suffix]+file_type)
                os.rename(file_old,file_new)
            elif file_wo_txt_suffix in change_txt_name.keys():
                file_old = os.path.join(root,file)
                file_new = os.path.join(root,change_txt_name[file_wo_txt_suffix]+txt_type)
                os.rename(file_old,file_new)
            # # add parameters to files
            # for folder in dirs:
            #     if folder in add_param_name.keys():
            #         new_params = add_param_name[folder]
            #         new_params = pd.DataFrame.from_dict(new_params)
            #         data =


