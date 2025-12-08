#! /usr/bin/env python

"""Stump script for debugging of CLI application, it is similar to the one
generated during installation of our package.

The script can be used by code IDE, by setting configuration in the `launch.json`
configuration file with arguments for features you wish to debug.
For example:
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Debug Installed Module",
            "type": "debugpy",
            "request": "launch",
            "console": "integratedTerminal",
            "program": "${workspaceFolder}/scripts/debug_runner.py",
            "args": [<URL_OF_A_LOG>]
        }
    ]
}
```"""
from logdetective.logdetective import main

if __name__ == "__main__":
    main()
