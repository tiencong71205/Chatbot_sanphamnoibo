import sys
import unittest

loader = unittest.TestLoader()
suite = loader.discover('/app/tests', pattern='test_*.py')
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

if not result.wasSuccessful():
    sys.exit(1)
