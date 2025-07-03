from __future__ import annotations
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective, SphinxRole
from sphinx.util.typing import ExtensionMetadata
from docutils.statemachine import ViewList
from sphinx.util.nodes import nested_parse_with_titles
import importlib
import pdb

"""
TO DUBUG:

import pdb;
pdb.set_trace()

The latter line can be placed anywhere in the code and will set a breakpoint
at the location. When calling "make html" to build the documentatin, the 
breakpoint will be activated. Type variable names to view variables values
"next" to go to the next line, "step" to step into a function, and "exit" to
leave the debugger.
"""

class DocstringMethod(SphinxDirective):
    """A directive insert the trimed docstring of a class method.

    The can be used to insert the docstring of a class method directly into 
    the documentation text. It removes the function header, summary line, 
    and the information about parameters, arguments, and return types. The 
    only text that is inserted into the documentation is therefore the main
    body of the original docstring.

    The directive is useful for minimizing the amount of redundant information
    that is required when writting both docstrings and documentation. For
    example, the mathematical formulation of constraints can now be written
    only once (in the docstring) and copied directly over to the documentation.

    To use this directive, use the following command in a restructured text 
    file:

    .. docstring_method <module.class.method_name>

    EXAMPLE

    .. docstring_method:: zen_garden.model.objects.technology.conversion_technology.ConversionTechnologyRules.constraint_capacity_factor_conversion

      
    """

    required_arguments = 1

    def run(self):
        full_name = self.arguments[0]
        module_name, _, obj_name = full_name.rpartition('.')

        try:
            parts = full_name.split(".")
            module_name = ".".join(parts[:-2] if len(parts) > 2 else parts[:-1])
            obj_path = parts[-2:] if len(parts) > 2 else parts[-1:]

            module = importlib.import_module(module_name)
            obj = module
            for attr in obj_path:
                obj = getattr(obj, attr)

            doc = obj.__doc__ or ""
            lines = doc.strip().splitlines()

            # Remove summary (first non-empty line)
            while lines and not lines[0].strip():
                lines.pop(0)
            if lines:
                lines.pop(0)  # Remove summary

            # Remove param/return sections
            filtered = []
            for line in lines:
                if line.strip().startswith((':param', ':return', 
                                            ':rtype', ':raises')):
                    break
                filtered.append(line)

            # Convert list of strings to a ViewList so Sphinx can parse it
            content = ViewList()
            for i, line in enumerate(filtered):
                content.append(line, f"{full_name} docstring", i)

            # Create a container node and parse the content into it
            node = nodes.section()
            node.document = self.state.document
            nested_parse_with_titles(self.state, content, node)

            return node.children
        
        except Exception as e:
            error = self.state_machine.reporter.error(
                f"Failed to extract docstring for '{full_name}': {e}",
                line=self.lineno)
            return [error]


class DocstringClass(SphinxDirective):
    """A directive insert the trimed docstring of a class.

    The can be used to insert the docstring of a class directly into 
    the documentation text. It removes the function header, summary line, 
    and the information about parameters, arguments, and return types. The 
    only text that is inserted into the documentation is therefore the main
    body of the original docstring.

    The directive is useful for minimizing the amount of redundant information
    that is required when writting both docstrings and documentation. For
    example, the docstring for conversion technologies can now contain all
    required information and pasted directly into the documentation.

    To use this directive, use the following command in a restructured text 
    file:

    .. docstring_class <module.class_name>

    EXAMPLE

    .. docstring_class:: zen_garden.model.objects.technology.conversion_technology.ConversionTechnologyRules

      
    """

    required_arguments = 1

    def run(self):
        full_name = self.arguments[0]
        module_name, _, obj_name = full_name.rpartition('.')

        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_name)

            doc = obj.__doc__ or ""
            lines = doc.strip().splitlines()

            # Remove summary (first non-empty line)
            while lines and not lines[0].strip():
                lines.pop(0)
            if lines:
                lines.pop(0)  # Remove summary

            # Remove param/return sections
            filtered = []
            for line in lines:
                if line.strip().startswith((':param', ':return', 
                                            ':rtype', ':raises')):
                    break
                filtered.append(line)

            # Convert list of strings to a ViewList so Sphinx can parse it
            content = ViewList()
            for i, line in enumerate(filtered):
                content.append(line, f"{full_name} docstring", i)

            # Create a container node and parse the content into it
            node = nodes.section()
            node.document = self.state.document
            nested_parse_with_titles(self.state, content, node)

            return node.children
        
        except Exception as e:
            error = self.state_machine.reporter.error(
                f"Failed to extract docstring for '{full_name}': {e}",
                line=self.lineno)
            return [error]

def setup(app: Sphinx) -> ExtensionMetadata:
    """
    Setup directives
    
    This function is required in order to register the directives with
    Sphynx. The name of the directives, as seen by the user, is set here.
    """
    app.add_directive('docstring_method', DocstringMethod)
    app.add_directive('docstring_class', DocstringClass)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }