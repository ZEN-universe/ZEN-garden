import pyomo.environ as pe

def input_carrier_rule(model):
    return([(tech,carrier) for tech in model.tech for carrier in model.input_carrier[tech]])

def output_carrier_rule(model):
    return([(tech,carrier) for tech in model.tech for carrier in model.output_carrier[tech]])

def constraint_conversion_rule(model,tech):
    return(
        sum(model.output_eff[tech,carrier]*model.output_flow[tech,carrier] for carrier in model.output_carrier[tech]) == sum(model.input_eff[tech,carrier]*model.input_flow[tech,carrier] for carrier in model.input_carrier[tech])
    )

def constraint_conversion2_rule(model,tech):
    return(
        sum(2*model.output_eff[tech,carrier]*model.output_flow[tech,carrier] for carrier in model.output_carrier[tech]) == -sum(model.input_eff[tech,carrier]*model.input_flow[tech,carrier] for carrier in model.input_carrier[tech])
    )

def main():
    model = pe.ConcreteModel()
    input_dict_raw = {}
    input_dict_raw["tech"] = ["PV", "CHP", "Wind"]
    input_dict_raw["carrier"] = ["sun","love","gas","wind","electricity","heat"]
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

    model.tech = pe.Set(initialize = input_dict_raw["tech"])
    model.carrier = pe.Set(initialize = input_dict_raw["carrier"])
    model.input_carrier = pe.Set(
        model.tech,
        initialize = input_dict_raw["input_carrier"]
    )
    model.output_carrier = pe.Set(
        model.tech,
        initialize = input_dict_raw["output_carrier"]
    )
    model.input_combi = pe.Set(dimen = 2,initialize = input_carrier_rule)
    model.output_combi = pe.Set(dimen = 2,initialize = output_carrier_rule)
    # Params
    model.input_eff = pe.Param(
        model.input_combi,
        within = pe.NonNegativeReals,
        initialize = input_dict_raw["input_eff"]
    )
    model.output_eff = pe.Param(
        model.output_combi,
        within = pe.NonNegativeReals,
        initialize = input_dict_raw["output_eff"]
    )
    # Vars
    model.input_flow = pe.Var(
        model.input_combi,
        within = pe.NonNegativeReals
    )
    model.output_flow = pe.Var(
        model.output_combi,
        within = pe.NonNegativeReals
    )
    # Constraints
    model.constraint_conversion_list = pe.ConstraintList()

    model.constraint_conversion = pe.Constraint(
        model.tech,
        rule = constraint_conversion_rule
    )
    model.constraint_conversion2 = pe.Constraint(
        model.tech,
        rule = constraint_conversion2_rule
    )
    model.constraint_conversion_list.add(model.constraint_conversion)
    model.constraint_conversion_list.add(model.constraint_conversion2)
    a=1
    

if __name__ == "__main__":
    main()