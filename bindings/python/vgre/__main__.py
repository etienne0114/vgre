"""Enable ``python -m vgre`` to invoke the same CLI as the ``vgre`` command."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
