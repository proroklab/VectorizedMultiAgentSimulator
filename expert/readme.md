## Using the expert

This program will load in the task logs in `log_dir` and use them to generate actions in the `navigation2` task.

Information about the arguments can be found by running `python3 NavigationExpert.py -h`

The grid_map.xml, config.xml, and task log files should all be taken from the CCBS fork. More information about these can be found there.

Make sure that the task log files are all named in the following manner: 

`<anything you want>_<a unique number>_log.xml`

Current assumptions: 
- Every file in `log_dir` that ends in `.xml` should be used
- All tasks use the same map
- The agents should definitely not collide. This leads to the agents driving relatively slowly. 
  - The `select_action` method can be modified to fix this but CCBS assumes all agents move at the same speed which is a big limiting factor. db-CBS may be able to help