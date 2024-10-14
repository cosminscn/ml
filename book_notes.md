# A Philosophy of Software Design - John Ousterhout

-  Modules should be deep

-  Information hiding (and leakage)
  - information leakage, temporal decomposition, overexposure (api)

-  General purpose modules are deeper
  - A `Text` class can have 2 methods instead of multiple, and the UI class can use these to solve problems.
  - A `History` class can keep track of actions which are specialized, and can add fences and group management.
  - Push complexity up or down
    - os drivers have the same interface but different behavior, dispatch method in this case, pytorch as well?

- Different layer, different abstraction
  - Passthrough methods indicate bad responsibility design.
  - Note: blenders or internalrequest
  - Use an immutable context instead of passthrough variables.
  

