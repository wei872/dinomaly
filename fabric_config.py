"""YAML defaults for the MobileNet CLI; explicit CLI arguments always win."""
from pathlib import Path

import yaml


def apply_config_defaults(parser, argv=None):
    preliminary, _ = parser.parse_known_args(argv)
    path = Path(preliminary.config)
    if not path.is_file():
        parser.error(f'配置文件不存在: {path}')
    with path.open(encoding='utf-8-sig') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get('pipeline') != 'fabric_mobilenet':
        parser.error('请使用MobileNet配置；旧DINO配置不能用于fabric_pipeline.py')
    if config.get('backbone') != 'mobilenet_v3_small_partial':
        parser.error('当前轻量入口只支持 mobilenet_v3_small_partial')
    defaults = {k: v for k, v in config.items() if k not in ('pipeline', 'backbone')}
    actions = {action.dest: action for action in parser._actions
               if action.dest not in ('mode', 'config', 'help')}
    unknown = defaults.keys() - actions.keys()
    if unknown:
        parser.error(f'未知配置项: {sorted(unknown)}；类别数/名称请只在dataset YAML中配置')
    for key, value in defaults.items():
        action = actions[key]
        if isinstance(action.default, bool) and not isinstance(value, bool):
            parser.error(f'{key} 必须为YAML布尔值 true/false')
        if action.type is not None:
            try:
                defaults[key] = action.type(value)
            except (ValueError, TypeError):
                parser.error(f'配置项 {key} 的类型不正确')
    if 'data' in defaults and not Path(defaults['data']).is_absolute():
        defaults['data'] = str((path.resolve().parent / defaults['data']).resolve())
    parser.set_defaults(**defaults)
