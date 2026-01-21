#!/usr/bin/env python3
"""
变更集功能演示脚本

演示知识图谱的版本控制功能：
1. 创建变更集
2. 添加新的三元组
3. 查看变更集列表
4. 查看审计事件
5. 回滚变更集
"""

import yaml
from pathlib import Path
from app.neo4j_client import Neo4jClient
from ingest.versioning import GraphVersioning


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_changesets(changesets: list):
    if not changesets:
        print("  (无变更集)")
        return
    for cs in changesets:
        status_icon = "✓" if cs.get("status") == "active" else "✗"
        print(f"  [{status_icon}] {cs.get('change_id', '')[:8]}...")
        print(f"      操作者: {cs.get('actor', 'N/A')}")
        print(f"      原因: {cs.get('reason', 'N/A')}")
        print(f"      事件数: {cs.get('event_count', 0)}")
        print(f"      状态: {cs.get('status', 'N/A')}")
        print(f"      创建时间: {cs.get('created_at', 'N/A')}")
        print()


def print_audit_events(events: list):
    if not events:
        print("  (无审计事件)")
        return
    for e in events:
        rolled_back = " [已回滚]" if e.get("rolled_back") else ""
        print(f"  - {e.get('action', 'N/A')}: ({e.get('head', '')}) -[{e.get('relation', '')}]-> ({e.get('tail', '')}){rolled_back}")
        print(f"    修订版本: {e.get('revision', 1)}, 时间: {e.get('timestamp', 'N/A')}")


def main():
    print("\n" + "="*60)
    print("       知识图谱变更集功能演示")
    print("="*60)

    config = load_config()
    neo4j_cfg = config.get("neo4j", {})

    client = Neo4jClient(
        uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
        user=neo4j_cfg.get("user", "neo4j"),
        password=neo4j_cfg.get("password", "")
    )

    versioning = GraphVersioning(client)

    # 步骤1: 查看现有变更集
    print_section("步骤1: 查看现有变更集")
    existing = versioning.list_changesets(limit=5)
    print_changesets(existing)

    # 步骤2: 创建新变更集
    print_section("步骤2: 创建新变更集")
    change_id = versioning.begin_changeset(
        actor="demo_user",
        reason="演示：添加高血压相关知识",
        source="demo_script"
    )
    print(f"  已创建变更集: {change_id}")

    # 步骤3: 添加三元组
    print_section("步骤3: 添加医学知识三元组")

    triples = [
        {
            "head": "高血压",
            "relation": "HAS_SYMPTOM",
            "tail": "头晕",
            "confidence": 0.95,
            "source": "医学教材"
        },
        {
            "head": "高血压",
            "relation": "HAS_SYMPTOM",
            "tail": "头痛",
            "confidence": 0.92,
            "source": "医学教材"
        },
        {
            "head": "高血压",
            "relation": "RECOMMENDED_DRUG",
            "tail": "硝苯地平",
            "confidence": 0.88,
            "source": "临床指南"
        },
        {
            "head": "高血压",
            "relation": "NEEDS_EXAM",
            "tail": "血压测量",
            "confidence": 0.99,
            "source": "诊疗规范"
        },
    ]

    for t in triples:
        versioning.upsert_triple(t, change_id)
        print(f"  + ({t['head']}) -[{t['relation']}]-> ({t['tail']})")

    print(f"\n  共添加 {len(triples)} 条三元组")

    # 步骤4: 查看审计事件
    print_section("步骤4: 查看变更集的审计事件")
    events = versioning.get_audit_events(change_id)
    print_audit_events(events)

    # 步骤5: 查看更新后的变更集列表
    print_section("步骤5: 查看更新后的变更集列表")
    updated = versioning.list_changesets(limit=5)
    print_changesets(updated)

    # 步骤6: 查看实体历史
    print_section("步骤6: 查看实体变更历史")
    history = versioning.get_entity_history("高血压")
    if history:
        for h in history:
            active_str = "活跃" if h.get("active") else "已失效"
            print(f"  - {h.get('relation', 'N/A')} (版本 {h.get('revision', 1)}) [{active_str}]")
            print(f"    变更集: {str(h.get('change_id', ''))[:8]}...")
            print(f"    生效时间: {h.get('valid_from', 'N/A')}")
    else:
        print("  (无历史记录)")

    # 步骤7: 演示回滚
    print_section("步骤7: 演示回滚功能")
    print(f"  是否回滚变更集 {change_id[:8]}...?")

    user_input = input("  输入 'y' 确认回滚, 其他键跳过: ").strip().lower()

    if user_input == 'y':
        count = versioning.rollback(change_id)
        print(f"  已回滚 {count} 个审计事件")

        print("\n  回滚后的变更集状态:")
        final = versioning.list_changesets(limit=5)
        print_changesets(final)

        print("\n  回滚后的审计事件:")
        events_after = versioning.get_audit_events(change_id)
        print_audit_events(events_after)
    else:
        print("  已跳过回滚演示")

    # 清理
    client.close()

    print_section("演示完成")
    print("  变更集功能可用于:")
    print("  - 追踪知识图谱的每次修改")
    print("  - 记录修改者和修改原因")
    print("  - 在发现错误时回滚到之前的状态")
    print("  - 审计知识来源和变更历史")
    print()


if __name__ == "__main__":
    main()
