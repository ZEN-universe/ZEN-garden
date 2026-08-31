.. _class_structure.class_structure:

Class Structure
===============

.. mermaid::
   :zoom:

   ---
   title: Elements
   ---
   classDiagram
       class Element {
           +own_parameters
           +own_sets
           +always_construct
           +_initialize()
           +prepare_input_data()
           +finalize_input_data()
       }

       class Technology {
           +prepare_input_data()
       }
       class Carrier
       class EnergySystem {
           +_initialize()
           +finalize_input_data()
       }
       class ConversionTechnology {
           +_initialize()
       }
       class StorageTechnology {
           +_initialize()
       }
       class RetrofittingTechnology {
           +prepare_input_data()
       }
       class TransportTechnology {
           +_initialize()
       }


       Element <|-- Technology
       Element <|-- Carrier
       Element <|-- EnergySystem
       Technology <|-- ConversionTechnology
       Technology <|-- StorageTechnology
       Technology <|-- TransportTechnology
       ConversionTechnology <|-- RetrofittingTechnology

Every ``Element`` subclass declares which sets, parameters, variables,
constraints, and expressions belong to it through class attributes
(``own_sets``, ``own_parameters``, ``variables``, ``constraints``,
``expressions``); see :ref:`adding_elements.structure`. Only the methods
actually overridden by a subclass are listed on it above; the rest are
inherited unchanged from ``Element``.


.. mermaid::
   :zoom:

   ---
   title: Model Construction
   ---
   classDiagram
       class ModelConstructor
       class ElementRegistry
       class OptimizationModel
       class ModelSchema
       class NetworkTopology
       class TimeStepsDicts

       ModelConstructor --> ElementRegistry
       ModelConstructor --> OptimizationModel
       ModelConstructor --> ModelSchema
       ModelConstructor --> NetworkTopology
       ModelConstructor --> TimeStepsDicts

There is one ``ModelConstructor`` instance per element *type* (not per
concrete element); it reads that type's declared sets, parameters, variables,
and constraints off the element class and builds them through the
collaborators shown above. No subclassing is needed to add a new element
type.


.. mermaid::
   :zoom:

   ---
   title: Default Config
   ---
   classDiagram

       class ConfigBase
       class Config
       class System
       class Solver
       class Analysis
       class Subsets
       class HeaderDataInputs
       class TimeSeriesAggregation

       ConfigBase <|-- Config
       ConfigBase <|-- Analysis
       ConfigBase <|-- Solver
       ConfigBase <|-- System
       ConfigBase <|-- Subsets
       ConfigBase <|-- HeaderDataInputs
       ConfigBase <|-- TimeSeriesAggregation
       Config *-- Analysis
       Config *-- Solver
       Config *-- System
       Analysis *-- Subsets
       Analysis *-- HeaderDataInputs
       Analysis *-- TimeSeriesAggregation


.. mermaid::
   :zoom:

   ---
   title: Optimization Model
   ---
   classDiagram
       class OptimizationModel
       class Registry
       class SetRegistry
       class ParameterRegistry
       class VariableRegistry
       class ConstraintRegistry
       class BaseSet
       class SimpleSet
       class IndexedSet
       class DictParameter
       class GenericSet
       class GenericParameter
       class GenericVariable
       class GenericConstraint

       Registry <|-- SetRegistry
       Registry <|-- ParameterRegistry
       Registry <|-- VariableRegistry
       Registry <|-- ConstraintRegistry
       BaseSet <|-- SimpleSet
       BaseSet <|-- IndexedSet
       SetRegistry *-- BaseSet
       ParameterRegistry *-- DictParameter
       OptimizationModel *-- SetRegistry
       OptimizationModel *-- ParameterRegistry
       OptimizationModel *-- VariableRegistry
       OptimizationModel *-- ConstraintRegistry
       GenericSet ..> SetRegistry : build()
       GenericParameter ..> ParameterRegistry : build()
       GenericVariable ..> VariableRegistry : build()
       GenericConstraint ..> ConstraintRegistry : build()

``OptimizationModel`` holds one registry per component kind, each built once
for the whole model. ``GenericSet``, ``GenericParameter``, ``GenericVariable``,
and ``GenericConstraint`` are the base classes a developer subclasses to add a
new component (see :ref:`adding_elements.structure` and :ref:`linopy.linopy`);
each subclass's ``build`` method populates the matching registry.


.. mermaid::
   :zoom:

   ---
   title: Other Classes
   ---
   classDiagram
       class IISConstraintParser
       class ScenarioDict
       class InputDataChecks
       class StringUtils
       class ScenarioUtils
       class OptimizationError
       class ElementDataLoader
       class TimeSeriesAggregation
       class UnitConverter
       class Scaling
       class DatasetPathResolver
       class ModelConstructionService
       class ElementFactory
       class MultiIndexHelper
