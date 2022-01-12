import pyomo.environ as pe

def input_carrier_rule(model):
    return([(tech,carrier) for tech in model.tech for carrier in model.input_carrier[tech]])

def output_carrier_rule(model):
    return([(tech,carrier) for tech in model.tech for carrier in model.output_carrier[tech]])


def main():
    model = pe.AbstractModel()
    model.tech = pe.Set()
    model.carrier = pe.Set()
    model.input_carrier = pe.Set(
        model.tech
    )
    model.output_carrier = pe.Set(
        model.tech
    )
    model.input_combi = pe.Set(dimen = 2,initialize = input_carrier_rule)
    model.output_combi = pe.Set(dimen = 2,initialize = output_carrier_rule)
    model.input_eff = pe.Param(
        model.input_combi,
        within = pe.NonNegativeReals
    )
    model.output_eff = pe.Param(
        model.output_combi,
        within = pe.NonNegativeReals
    )

    input_dict_raw = {}
    input_dict_raw["tech"] = {None:["PV", "CHP", "Wind"]}
    input_dict_raw["carrier"] = {None:["sun","love","gas","wind","electricity","heat"]}
    input_dict_raw["input_carrier"] = { "PV":["sun","love"],
                                        "CHP": ["gas"],
                                        "Wind": ["wind"]
                                    }
    input_dict_raw["output_carrier"] = {"PV":["electricity"],
                                        "CHP": ["electricity","heat"],
                                        "Wind": ["electricity"]
                                    }
    input_dict_raw["input_eff"] = { ('PV', 'sun'):1, 
                                    ('PV', 'love'):5, 
                                    ('CHP', 'gas'):0.3, 
                                    ('Wind', 'wind'):0.1}
    input_dict_raw["output_eff"] = { ('PV', 'electricity'):0.1, 
                                    ('CHP', 'electricity'):0.4, 
                                    ('CHP', 'heat'):0.3, 
                                    ('Wind', 'electricity'):1.2}
                                                 

    instance = model.create_instance({None: input_dict_raw})
    a=1
if __name__ == "__main__":
    main()