"""
Created on 2022/06/07
@author Sangwoo Han
"""
import click


@click.group()
@click.option("--save-args", type=click.Path(), help="Save command args")
@click.pass_context
def cli(ctx: click.core.Context, save_args):
    ctx.ensure_object(dict)
    ctx.obj['save_args'] = save_args


if __name__ == "__main__":
    from hp_tuning import *
    from train import *
    from preprocess import *

    cli()
