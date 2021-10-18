    @staticmethod
    def constraint_max_carrier_import_rule(model, carrier: str, node: int, time: int) -> 'pyomo rule':
        '''
        :param model:
        :param carrier:
        :param node:
        :param time:
        :return:
        '''

        return(
                    model.importCarrier[carrier, node, time] <= model.gridIn[carrier, node, time]
            )

    def product_balance_rule(model, carrier, node, time):
        """

        :param model:
        :param carrier:
        :param node:
        :param time:
        :return:
        """

        return(
                model.demand[carrier, node, time] - model.supply[carrier, node, time] - sum(model.production[carrier,node,time,tech] for tech in model.setTechnologies) == 0
        )

        return(eval(textfile))