"""
fetch_api_stats.py
==================
API-Football から EPL の試合スタッツを取得して JSON として保存するスクリプト。
ローカルPCで実行し、出力 JSON を GitHub にコミットしてください。

使い方:
  1. API-Football のAPIキーを環境変数に設定
     export APIFOOTBALL_KEY="あなたのAPIキー"

  2. 取得するシーズンを指定して実行
     python fetch_api_stats.py --season 2024    # 2024/25 シーズン
     python fetch_api_stats.py --season 2023    # 2023/24 シーズン
     python fetch_api_stats.py --season 2022    # 2022/23 シーズン

  3. 出力ファイル api_stats_XXXX-XX.json を GitHub リポジトリにコミット

注意:
  - 無料プランは2022〜2024のみ対応 (2025は有料プランが必要)
  - 380試合 × 1リクエスト = 380リクエスト（100req/日制限）
  - 自動的に進捗を保存するので、中断しても翌日再実行で続きから取得できます
  - 全取得に最低4日かかります（100req/日）
"""

import argparse
import json
import os
import sys
import time

import requests

# ── 設定 ─────────────────────────────────────────────────────────────
APF_BASE   = "https://v3.football.api-sports.io"
EPL_LEAGUE = 39
SEASON_MAP = {
    2025: "2025-26",   # 有料プランのみ
    2024: "2024-25",
    2023: "2023-24",
    2022: "2022-23",
}


def apf_get(endpoint: str, params: dict, api_key: str) -> dict | None:
    """API-Football への GET リクエスト"""
    headers = {
        "x-apisports-key": api_key,
        "Accept": "application/json",
    }
    try:
        r = requests.get(
            f"{APF_BASE}/{endpoint}",
            headers=headers,
            params=params,
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            # エラーチェック
            errors = data.get("errors", {})
            if errors:
                print(f"  API Error: {errors}")
                return None
            return data
        else:
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
            return None
    except Exception as e:
        print(f"  Request error: {e}")
        return None


def get_fixture_ids(season: int, api_key: str) -> list[int]:
    """指定シーズンの終了済み全試合IDを取得"""
    print(f"取得中: EPL {season}/{season+1} 試合一覧...")
    data = apf_get("fixtures", {"league": EPL_LEAGUE, "season": season}, api_key)
    if not data:
        return []
    finished = {"FT", "AET", "PEN"}
    ids = [
        f["fixture"]["id"]
        for f in data.get("response", [])
        if f.get("fixture", {}).get("status", {}).get("short") in finished
    ]
    print(f"  → 終了済み試合: {len(ids)} 件")
    return ids


def parse_fixture_stats(response: list) -> dict:
    """fixture/statistics レスポンスを {home:{...}, away:{...}} に変換"""
    result = {}
    for i, team_data in enumerate(response[:2]):
        side = "home" if i == 0 else "away"
        tname = team_data.get("team", {}).get("name", "")
        stats = {"team_name": tname}
        for stat in team_data.get("statistics", []):
            val = stat.get("value")
            t   = stat.get("type", "")
            if isinstance(val, str) and val.endswith("%"):
                try: val = float(val.rstrip("%"))
                except: val = None
            elif val is not None:
                try: val = float(val)
                except: pass
            stats[t] = val
        result[side] = stats
    return result


def fetch_all_stats(
    season: int,
    api_key: str,
    output_path: str,
    max_per_run: int = 95,
) -> None:
    """
    全試合のスタッツを取得して JSON に保存。
    中断しても output_path の内容を引き継いで続きから再開できる。
    """
    season_str = SEASON_MAP.get(season, f"{season}-{str(season+1)[-2:]}")

    # 既存キャッシュを読み込み（中断再開用）
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
                print(f"既存キャッシュ読み込み: {len(existing)} 試合")
            except Exception:
                pass

    # 試合ID一覧を取得
    fixture_ids = get_fixture_ids(season, api_key)
    if not fixture_ids:
        print("試合IDの取得に失敗しました。APIキーとシーズンを確認してください。")
        sys.exit(1)

    # 未取得の試合だけ取得
    missing = [fid for fid in fixture_ids if str(fid) not in existing]
    to_fetch = missing[:max_per_run]

    print(f"\n取得状況: {len(existing)}/{len(fixture_ids)} 済み")
    print(f"今回取得: {len(to_fetch)} 試合 (残り: {len(missing)} 試合)")

    if not to_fetch:
        print("✅ 全試合取得済みです！")
        return

    print("\n取得開始...")
    fetched = 0
    errors  = 0

    for i, fid in enumerate(to_fetch):
        # 残リクエスト確認（10試合ごと）
        if i % 10 == 0 and i > 0:
            status = apf_get("status", {}, api_key)
            if status and "response" in status:
                req = status["response"].get("requests", {})
                used  = req.get("current", "?")
                limit = req.get("limit_day", 100)
                remain = int(limit) - int(used) if str(used).isdigit() else "?"
                print(f"  [{i}/{len(to_fetch)}] 残りリクエスト: {remain}/日")

                # 残り5以下なら停止
                if isinstance(remain, int) and remain <= 5:
                    print("⚠️  本日のリクエスト上限に近づきました。停止します。")
                    print("   明日また実行してください。")
                    break

        print(f"  [{i+1:3d}/{len(to_fetch)}] fixture {fid}", end="", flush=True)
        data = apf_get("fixtures/statistics", {"fixture": fid}, api_key)

        if data and data.get("response"):
            existing[str(fid)] = parse_fixture_stats(data["response"])
            fetched += 1
            print(f" ✓")
        else:
            errors += 1
            print(f" ✗ (スキップ)")

        # 中間保存（10試合ごと）
        if (i + 1) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False)

        time.sleep(0.4)  # レート制限対応

    # 最終保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)

    total = len(existing)
    print(f"\n✅ 保存完了: {output_path}")
    print(f"   今回取得: {fetched} 試合 / エラー: {errors} 試合")
    print(f"   合計保存: {total}/{len(fixture_ids)} 試合")

    if total < len(fixture_ids):
        remaining_days = (len(fixture_ids) - total + max_per_run - 1) // max_per_run
        print(f"\n   残り {len(fixture_ids) - total} 試合 → あと約 {remaining_days} 日で完了")
        print(f"   明日また実行してください: python fetch_api_stats.py --season {season}")
    else:
        print(f"\n🎉 全 {len(fixture_ids)} 試合の取得が完了しました！")
        print(f"   {output_path} を GitHub リポジトリにコミットしてください")


def check_status(api_key: str) -> None:
    """APIキーの状態を確認"""
    print("APIキーの状態を確認中...")
    data = apf_get("status", {}, api_key)
    if not data or "response" not in data:
        print("❌ APIキーが無効、または接続エラーです")
        return
    resp = data["response"]
    sub  = resp.get("subscription", {})
    req  = resp.get("requests", {})
    print(f"✅ プラン: {sub.get('plan', '不明')}")
    print(f"   有効期限: {sub.get('end', '不明')}")
    print(f"   本日使用: {req.get('current', 0)} / {req.get('limit_day', 100)} リクエスト")
    print(f"   残り: {int(req.get('limit_day', 100)) - int(req.get('current', 0))} リクエスト")


def main():
    parser = argparse.ArgumentParser(
        description="API-Football から EPL スタッツを取得して JSON に保存"
    )
    parser.add_argument(
        "--season", type=int, default=2024,
        help="シーズン開始年 (例: 2024 → 2024/25シーズン) ※無料プランは2022〜2024のみ"
    )
    parser.add_argument(
        "--max", type=int, default=95,
        help="1回の実行での最大取得数 (デフォルト: 95)"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="出力JSONファイルパス (デフォルト: api_stats_{season}-{season+1}.json)"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="APIキーの状態確認のみ"
    )
    args = parser.parse_args()

    # APIキー取得
    api_key = os.environ.get("APIFOOTBALL_KEY", "")
    if not api_key:
        print("❌ 環境変数 APIFOOTBALL_KEY が設定されていません")
        print("   export APIFOOTBALL_KEY='あなたのAPIキー'")
        sys.exit(1)

    if args.check:
        check_status(api_key)
        return

    season_str = SEASON_MAP.get(args.season, f"{args.season}-{str(args.season+1)[-2:]}")
    output = args.output or f"api_stats_{season_str}.json"

    print(f"=== API-Football EPL スタッツ取得 ===")
    print(f"  シーズン: {season_str}")
    print(f"  出力先:   {output}")
    print(f"  最大取得: {args.max} 試合/回")
    print()

    fetch_all_stats(args.season, api_key, output, max_per_run=args.max)

    print()
    print("=== 次のステップ ===")
    print(f"  git add {output}")
    print(f"  git commit -m 'Add API stats for {season_str}'")
    print(f"  git push")
    print()
    print("GitHub にコミット後、Streamlit Secrets に以下を追加:")
    print("  GITHUB_USER = 'あなたのGitHubユーザー名'")
    print("  GITHUB_REPO = 'リポジトリ名'")


if __name__ == "__main__":
    main()
