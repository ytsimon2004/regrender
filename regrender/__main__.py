from argclz import AbstractParser
from argclz.commands import parse_command_args

from .main_probe import ProbeOptions
from .main_register import RegisterOptions
from .main_roi import RoiOptions


class SetupOptions(AbstractParser):
    DESCRIPTION = ('Pre-download the Allen CCF atlas so the first register/roi/probe run '
                   'does not stall on a multi-minute download.')

    def run(self):
        from brainglobe_atlasapi import BrainGlobeAtlas

        from ._app import printf

        name = 'allen_mouse_10um'  # the resolution the app is built around (roi/probe assume 10 µm)
        printf(f'fetching {name} (one-time, may take a few minutes)...')
        BrainGlobeAtlas(name)  # downloads + caches if missing, no-op if already local
        printf(f'{name} ready')


def main():
    parse_command_args(
        usage='python -m regrender ...',
        description='regrender cli usage',
        parsers=dict(
            register=RegisterOptions,
            probe=ProbeOptions,
            roi=RoiOptions,
            setup=SetupOptions
        )
    )


if __name__ == '__main__':
    main()
