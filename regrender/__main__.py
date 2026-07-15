from argclz.commands import parse_command_args

from .main_probe import ProbeOptions
from .main_register import RegisterOptions
from .main_roi import RoiOptions


def main():
    parse_command_args(
        usage='python -m regrender ...',
        description='regrender cli usage',
        parsers=dict(
            register=RegisterOptions,
            probe=ProbeOptions,
            roi=RoiOptions
        )
    )
