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