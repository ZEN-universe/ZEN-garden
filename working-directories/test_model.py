import calliope

# increase logging verbosity
calliope.set_log_verbosity('INFO', include_solver_output=False)

# init model 
model = calliope.Model('working_directory_test_HSC/model.yaml')

model.run()
