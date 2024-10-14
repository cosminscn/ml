# A Philosophy of Software Design - John Ousterhout

- Modules should be deep

- Information hiding (and leakage)
  - Information leakage, temporal decomposition, overexposure (api)
  - Separate general behavior from specific behavior (maybe in different classes?)

- General purpose modules are deeper
  - A `Text` class can have 2 methods instead of multiple, and the UI class can use these to solve problems.
  - A `History` class can keep track of actions which are specialized, and can add fences and group management.
  - Push complexity up or down
    - OS drivers have the same interface but different behavior (dispatch method), similar to PyTorch's abstraction.

- Different layer, different abstraction
  - Passthrough methods indicate bad responsibility design.
  - Example: blenders or internal request handling.
  - Use an immutable context instead of passthrough variables.

- Pull complexity downwards
  - Config classes is an antipattern, try and figure out the default params
  - Make it easy for the user even if it means it's a bit harder for the developer

