# Energy-Carbon-Optimization-Platform

Optimization model of European energy systems and value chains. It includes the electricity system, hydrogen value chains, and carbon capture, storage and utilization (CCUS) value chains. 

## Modifications to the code
Multiple developers are working in parallel. To allow a smooth cooperation, the following procedures should be followed:
* any modification should be done in the "development" branch. If it doesn't exist, it should be created from the "master" branch
* for any modification, every developer should create a new branch of "development" to modify the code. 
* this can be done from the "development" branch with: 'git branch new_branch_name'
* once the modifications are done: 
  '''
  git add .
  git commit
  'it push origin new_branch_name
  '''
