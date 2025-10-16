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

       class Technology
       class Carrier
       class ConversionTechnology
       class StorageTechnology
       class RetrofittingTechnology
       class TransportTechnology


       Element <|-- Technology
       Element <|-- Carrier
       Technology <|-- ConversionTechnology
       Technology <|-- StorageTechnology
       Technology <|-- TransportTechnology
       ConversionTechnology <|-- RetrofittingTechnology


.. mermaid::
   :zoom:

   ---
   title: Rules
   ---
   classDiagram
       class GenericRule
       class CarrierRules
       class TechnologyRules
       class ConversionTechnologyRules
       class StorageTechnologyRules
       class RetrofittingTechnologyRules
       class TransportTechnologyRules

       GenericRule <|-- CarrierRules
       GenericRule <|-- TechnologyRules
       GenericRule <|-- ConversionTechnologyRules
       GenericRule <|-- StorageTechnologyRules
       GenericRule <|-- TransportTechnologyRules
       GenericRule <|-- RetrofittingTechnologyRules

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
       class ZenIndex
       class ZenSet
       class IndexSet
       class DictParameter
       class Parameter 
       class Variable 
       class Constraint

       Component <|-- IndexSet
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