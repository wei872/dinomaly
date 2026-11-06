"""Non-destructive, dependency-free cross-split exact-file duplicate audit."""
import hashlib
import json
from pathlib import Path


def audit_duplicates(split_paths, report_path):
    groups = {}
    for split, paths in split_paths.items():
        for path in paths:
            path = Path(path)
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            groups.setdefault(digest.hexdigest(), []).append(
                {'split': split, 'path': str(path.resolve())})
    duplicates = [{'sha256': digest, 'files': files} for digest, files in groups.items()
                  if len({item['split'] for item in files}) > 1]
    report = {'duplicate_group_count': len(duplicates),
              'duplicates': duplicates,
              'note': '仅检查文件内容完全相同的跨集合重复；不修改文件，不检测增强/重新编码近重复。'}
    report_path = Path(report_path)
    # Never allow a report to overwrite an input image.
    inputs = {Path(path).resolve() for paths in split_paths.values() for path in paths}
    if report_path.resolve() in inputs:
        raise ValueError('重复报告不能覆盖数据图片')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def filter_cross_split_duplicates(split_paths, report):
    """Keep training copies; exclude overlapping validation copies in memory only.

    If only validation splits overlap, prefer normalval (threshold calibration).
    Copies shared by the two training roles are retained in both roles.
    """
    skipped = []
    excluded = set()
    for group in report['duplicates']:
        files = group['files']
        training = any(item['split'] in ('normaltrain', 'train') for item in files)
        keep_splits = {'normaltrain', 'train'} if training else {'normalval'}
        for item in files:
            if item['split'] not in keep_splits:
                excluded.add((item['split'], str(Path(item['path']).resolve())))
                skipped.append(dict(item, reason='cross_split_duplicate', sha256=group['sha256']))
    filtered = {split: [p for p in paths if (split, str(Path(p).resolve())) not in excluded]
                for split, paths in split_paths.items()}
    return filtered, skipped
