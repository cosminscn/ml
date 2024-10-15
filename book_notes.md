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
 
- Better together or better apart
  - Together: shared info, shared usage, conceptual overlap, hard to understand them separately, code duplication (case x)
    - goto used to escape nested code and dedup short passage logging 
  - Apart: general purpose vs special purpose
  - Red flag: special general mixture
  - Red flag: conjoined methods

- Define errors out of existance
  - Example: substring, file deletion linux
  - Mask exceptions: nfs server retries
  - Exception aggregation/promotion: single place or single mechanism for handling exception
 
- Design it twice

- TODO: comments, naming, perf etc

# Simple Testing Can Prevent Most Critical Failures: An Analysis of Production Failures in Distributed Data-Intensive Systems
https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-yuan.pdf

- enforce code reviews on error-handling code, since the error handling logic is often simply wrong; 
