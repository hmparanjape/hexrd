from __future__ import print_function, division, absolute_import


descr = 'Cakes the 2D diffraction data'
example = """
examples:
    hexrd cake-data config.yml
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('cake-data', description = descr, help = descr)

    p.add_argument(
      'yml', type=str,
      help='YAML configuration file'
    )

    p.set_defaults(func=execute)


def execute(args, parser):
    import unittest

    suite = unittest.TestLoader().discover('hexrd')
    unittest.TextTestRunner(verbosity = args.verbose + 1).run(suite)
