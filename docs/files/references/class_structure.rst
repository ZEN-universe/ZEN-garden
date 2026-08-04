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
           +type1 attribute1
           +returntype1 method1()
       }

       class Technology {
           +store_input_data()
       }
       class Carrier {
           +store_input_data()
       }
       class ConversionTechnology {
           +store_input_data()
       }
       class StorageTechnology {
           +store_input_data()
       }
       class RetrofittingTechnology {
           +store_input_data()
       }
       class TransportTechnology {
           +store_input_data()
       }


       Element <|-- Technology
       Element <|-- Carrier
       Technology <|-- ConversionTechnology
       Technology <|-- StorageTechnology
       Technology <|-- TransportTechnology
       ConversionTechnology <|-- RetrofittingTechnology


.. mermaid::
   :zoom:

   ---
   title: Model Constructors
   ---
   classDiagram
       class ModelConstructor
       class CarrierConstructor
       class TechnologyConstructor
       class ConversionTechnologyConstructor
       class StorageTechnologyConstructor
       class RetrofittingTechnologyConstructor
       class TransportTechnologyConstructor

       ModelConstructor <|-- CarrierConstructor
       ModelConstructor <|-- TechnologyConstructor
       ModelConstructor <|-- ConversionTechnologyConstructor
       ModelConstructor <|-- StorageTechnologyConstructor
       ModelConstructor <|-- TransportTechnologyConstructor
       ModelConstructor <|-- RetrofittingTechnologyConstructor


.. mermaid::
   :zoom:

   ---
   title: Default Config
   ---
   classDiagram

       class Subscriptable
       class Config
       class System
       class Solver
       class Analysis
       class Subsets
       class HeaderDataInputs
       class TimeSeriesAggregation

       Subscriptable <|-- Config
       Config *-- Analysis
       Config *-- Solver
       Config *-- System
       Subscriptable <|-- Analysis
       Subscriptable <|-- TimeSeriesAggregation
       Subscriptable <|-- Solver
       Subscriptable <|-- System
       Subscriptable <|-- HeaderDataInputs
       Subscriptable <|-- Subsets


.. mermaid::
   :zoom:

   ---
   title: Components
   ---
   classDiagram
       class Component
       class MultiIndexHelper
       class ZenSet
       class SetRegistry
       class DictParameter
       class Parameter
       class Variable
       class Constraint

       Component <|-- SetRegistry
       Component <|-- Parameter
       Component <|-- Variable
       Component <|-- Constraint


..
   :zoom:


   ---
   title: Other Classes
   ---
   classDiagram
       class ISSConstraintParser
       class ScenarioDict
       class InputDataChecks
       class StringUtils
       class ScenarioUtils
       class OptimizationError
       class DataInput
       class TimeSeriesAggregation
       class TimeSteps
       class EnergySystem
       class UnitHandling
       class Scaling
       class ZenModel
       class DatasetPathResolver
       class ModelConstructionService
       class ElementRegistry
       class YearSpecificTs
