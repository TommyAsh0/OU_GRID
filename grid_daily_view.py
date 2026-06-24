"""
grid_daily_view.py — 动态网格日线级可视化工具 (pyecharts / HTML)

功能：
- 主图：日线K线 + MA + 动态网格线 + 开/平/强平点
- 副图：每日持仓层数 或 每日资金占用
- 输出HTML，适合回测研究与汇报展示

用法示例：

1) IDE/Python 直接运行（无命令行参数）：使用脚本内默认配置
2) 命令行运行（参数可覆盖默认值）：
    python grid_daily_view.py --code sh600452 --daily_csv ./output/daily_with_grid.csv --trades_csv ./output/trade_records.csv --summary_csv ./output/daily_summary.csv --max_grid 3
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Grid, Kline, Line, Scatter
from pyecharts.globals import CurrentConfig

CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/npm/echarts@latest/dist/"


OPEN_ACTIONS = {"OPEN", "OPEN_LONG"}
CLOSE_ACTIONS = {"CLOSE", "CLOSE_LONG"}
SHORT_OPEN_ACTIONS = {"OPEN_SHORT"}
SHORT_CLOSE_ACTIONS = {"CLOSE_SHORT"}
FORCE_CLOSE_ACTIONS = {"FORCE_CLOSE"}
BASE_DIR = Path(__file__).resolve().parent

# IDE 直接运行时使用的默认配置
# 新框架 bt_grid.py 输出文件名带有标的代码后缀，例如 daily_sh600452.csv
_DEFAULT_CODE = "sz300782"
DEFAULT_RUN_CONFIG = {
    "code": _DEFAULT_CODE,
    "daily_csv": str(BASE_DIR / "output_data" / f"daily_{_DEFAULT_CODE}.csv"),
    "trades_csv": str(BASE_DIR / "output_data" / f"trade_records_{_DEFAULT_CODE}.csv"),
    "summary_csv": str(BASE_DIR / "output_data" / f"daily_summary_{_DEFAULT_CODE}.csv"),
    "holding_stats_csv": str(BASE_DIR / "output_data" / f"holding_stats_{_DEFAULT_CODE}.csv"),
    "result_summary_csv": str(BASE_DIR / "output_data" / "result_summary.csv"),
    "max_grid": 2,
}


def _fmt_value(v):
    """统一格式化指标值: 浮点保留 4 位有效数, 百分比指标自动转 %, NaN → '-' """
    if v is None:
        return "-"
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return "-"
    except Exception:
        pass
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        if abs(v) >= 1:
            return f"{v:.4f}"
        return f"{v:.6f}"
    return str(v)


def _build_sidebar_html(code: str,
                        result_summary_csv: str = None,
                        holding_stats_csv: str = None) -> str:
    """构建右侧指标面板 HTML 片段 (含内联 CSS)��"""
    blocks = []

    # ----- 1) result_summary (本标的整行) -----
    if result_summary_csv and Path(result_summary_csv).exists():
        try:
            df = pd.read_csv(result_summary_csv)
            row = None
            if "ts_code" in df.columns and code in df["ts_code"].astype(str).values:
                row = df[df["ts_code"].astype(str) == code].iloc[0]
            elif len(df) > 0:
                row = df.iloc[0]
            if row is not None:
                rows_html = "".join(
                    f"<tr><td class='k'>{k}</td><td class='v'>{_fmt_value(v)}</td></tr>"
                    for k, v in row.items()
                )
                blocks.append(
                    "<div class='card'>"
                    "<div class='card-title'>📊 回测总览 (result_summary)</div>"
                    f"<table class='kv'>{rows_html}</table>"
                    "</div>"
                )
        except Exception as e:
            blocks.append(f"<div class='card'><b>读取 result_summary 失败</b>: {e}</div>")

    # ----- 2) holding_stats (按层) -----
    if holding_stats_csv and Path(holding_stats_csv).exists():
        try:
            df = pd.read_csv(holding_stats_csv)
            # 只展示有交易的层, 并按 side 分组
            df_show = df[df["trade_count"] > 0].copy() if "trade_count" in df.columns else df.copy()
            # 选要展示的列 (用户要求: trade_count, avg_holding_days_equivalent, win_rate)
            wanted_map = {
                "side": "方向",
                "level": "层级",
                "trade_count": "交易数",
                "avg_holding_days_equivalent": "平均持仓(天)",
                "win_rate": "平均胜率",
                "total_pnl": "总盈亏"
            }
            cols = [c for c in wanted_map.keys() if c in df_show.columns]

            def _side_label(s):
                try:
                    return "多" if int(s) > 0 else "空"
                except Exception:
                    return str(s)

            head = "".join(f"<th>{wanted_map[c]}</th>" for c in cols)
            body_rows = []
            for _, r in df_show.iterrows():
                cells = []
                for c in cols:
                    val = r[c]
                    if c == "side":
                        cells.append(f"<td>{_side_label(val)}</td>")
                    elif c == "win_rate":
                        try:
                            cells.append(f"<td>{float(val) * 100:.1f}%</td>"
                                         if pd.notna(val) else "<td>-</td>")
                        except Exception:
                            cells.append("<td>-</td>")
                    else:
                        cells.append(f"<td>{_fmt_value(val)}</td>")
                body_rows.append("<tr>" + "".join(cells) + "</tr>")

            blocks.append(
                "<div class='card'>"
                "<div class='card-title'>📈 层级仓位统计 (holding_stats)</div>"
                f"<table class='grid-table'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(body_rows)}</tbody></table>"
                "</div>"
            )
        except Exception as e:
            blocks.append(f"<div class='card'><b>读取 holding_stats 失败</b>: {e}</div>")

    if not blocks:
        return ""

    css = """
    <style>
      html, body { 
        margin: 0; padding: 0; background-color: #ffffff; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
      }
      #page-flex-root { display: flex; flex-direction: row; height: 100vh; overflow: hidden; }
      .chart-wrap { flex: 1 1 auto; min-width: 0; height: 100vh; overflow: hidden; position: relative; }
      .metrics-sidebar {
        width: 480px;
        min-width: 480px;
        height: 100vh;
        overflow-y: auto;
        padding: 24px;
        font-size: 13px;
        color: #1f2937;
        background: #f8fafc;
        border-left: 1px solid #e2e8f0;
        box-sizing: border-box;
        box-shadow: -4px 0 15px -3px rgba(0, 0, 0, 0.05);
      }
      /* Custom Scrollbar for sidebar */
      .metrics-sidebar::-webkit-scrollbar { width: 6px; }
      .metrics-sidebar::-webkit-scrollbar-track { background: transparent; }
      .metrics-sidebar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
      
      .metrics-sidebar h2 {
        font-size: 18px; font-weight: 700; margin: 0 0 20px 0;
        color: #0f172a; 
        display: flex; align-items: center; gap: 8px;
        letter-spacing: -0.025em;
      }
      
      .metrics-sidebar .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      }
      
      .metrics-sidebar .card-title {
        font-weight: 600; font-size: 14px;
        margin-bottom: 12px; color: #334155;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 8px;
        display: flex; align-items: center; gap: 6px;
      }
      
      /* kv table */
      .metrics-sidebar table.kv { border-collapse: collapse; width: 100%; }
      .metrics-sidebar table.kv tr { border-bottom: 1px solid #f1f5f9; }
      .metrics-sidebar table.kv tr:last-child { border-bottom: none; }
      .metrics-sidebar table.kv tr:hover td { background-color: #f8fafc; }
      .metrics-sidebar table.kv td { padding: 8px 4px; vertical-align: middle; }
      .metrics-sidebar table.kv td.k {
        color: #64748b; width: 55%; font-weight: 500;
      }
      .metrics-sidebar table.kv td.v {
        color: #0f172a; text-align: right;
        font-variant-numeric: tabular-nums; font-weight: 600;
      }
      
      /* grid-table */
      .metrics-sidebar table.grid-table {
        border-collapse: separate; border-spacing: 0; width: 100%; font-size: 12px;
        border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;
      }
      .metrics-sidebar table.grid-table th {
        background: #f1f5f9; color: #475569; font-weight: 600;
        padding: 8px; text-align: right; border-bottom: 1px solid #e2e8f0;
      }
      .metrics-sidebar table.grid-table td {
        padding: 8px; border-bottom: 1px solid #f1f5f9;
        text-align: right; font-variant-numeric: tabular-nums; color: #1e293b;
      }
      .metrics-sidebar table.grid-table tr:last-child td { border-bottom: none; }
      .metrics-sidebar table.grid-table th:first-child,
      .metrics-sidebar table.grid-table td:first-child,
      .metrics-sidebar table.grid-table th:nth-child(2),
      .metrics-sidebar table.grid-table td:nth-child(2) { text-align: center; }
      .metrics-sidebar table.grid-table tr:hover td { background: #f8fafc; }
      
      /* Tag styling for Side */
      .tag-long { background: #dcfce7; color: #0f5132; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;}
      .tag-short { background: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;}
    </style>
    """
    
    # 替换原本简单的 '多' / '空' 文本为有颜色的 Tag
    blocks_html = "".join(blocks)
    blocks_html = blocks_html.replace('<td>多</td>', '<td><span class="tag-long">多</span></td>')
    blocks_html = blocks_html.replace('<td>空</td>', '<td><span class="tag-short">空</span></td>')
    
    return (css
            + "<div class='metrics-sidebar'>"
            + f"<h2>📊 {code} 量化指标看板</h2>"
            + blocks_html
            + "</div>")


def _inject_sidebar(html_path: Path, sidebar_html: str) -> None:
    """把侧边栏 HTML 注入到 pyecharts 生成的 HTML 里。"""
    if not sidebar_html:
        return
    import re as _re
    text = html_path.read_text(encoding="utf-8")

    # 1) 兼容 pyecharts 生成的 <body > (带空格) / <body>
    body_open_pat  = _re.compile(r'(<body[^>]*>)', _re.IGNORECASE)
    body_close_pat = _re.compile(r'(</body>)',      _re.IGNORECASE)
    if not body_open_pat.search(text) or not body_close_pat.search(text):
        print("warning: <body> tag not found in rendered HTML, sidebar skipped.")
        return

    # 3) <body…> 后插入 flex 容器开口
    flex_open = (
        '\n<div id="page-flex-root">'
        '<div class="chart-wrap">'
    )
    def _repl_open(m):
        return m.group(1) + flex_open
    text = body_open_pat.sub(_repl_open, text, count=1)

    # 4) </body> 前插入侧边栏 + 关闭 flex 容器
    def _repl_close(m):
        return '</div>\n' + sidebar_html + '\n</div>\n' + m.group(1)
    text = body_close_pat.sub(_repl_close, text, count=1)

    html_path.write_text(text, encoding="utf-8")


def render_daily_grid_chart(
    code: str,
    daily_csv: str,
    trades_csv: str,
    summary_csv: str = None,
    max_grid: int = 3,
    holding_stats_csv: str = None,
    result_summary_csv: str = None,
):
    # =========================
    # 读取日线 + 网格数据
    # =========================
    df_daily = pd.read_csv(daily_csv)
    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"])
    df_daily = df_daily.sort_values("trade_date").reset_index(drop=True)

    if df_daily.empty:
        print("daily_csv is empty.")
        return

    # 自动从列名推断 max_grid（兼容多/空网格新格式）
    import re
    auto_n = 0
    pat = re.compile(r'^grid_(?:buy|sell)_(\d+)$')
    for c in df_daily.columns:
        m = pat.match(c)
        if m:
            auto_n = max(auto_n, int(m.group(1)))
    if auto_n > 0:
        max_grid = max(max_grid, auto_n)

    # =========================
    # 读取交易记录
    # =========================
    df_trades = pd.read_csv(trades_csv)
    if "trade_date" in df_trades.columns:
        df_trades["trade_date"] = pd.to_datetime(df_trades["trade_date"])
    elif "datetime" in df_trades.columns:
        df_trades["datetime"] = pd.to_datetime(df_trades["datetime"])
        df_trades["trade_date"] = df_trades["datetime"].dt.normalize()
    else:
        raise ValueError("trades_csv 必须包含 trade_date 或 datetime 列")

    # =========================
    # 读取每日汇总
    # =========================
    df_summary = None
    if summary_csv:
        df_summary = pd.read_csv(summary_csv)
        df_summary["trade_date"] = pd.to_datetime(df_summary["trade_date"])
        df_summary = df_summary.sort_values("trade_date").reset_index(drop=True)

    # =========================
    # X轴
    # =========================
    x_data = df_daily["trade_date"].dt.strftime("%Y-%m-%d").tolist()
    ts_index = {t: i for i, t in enumerate(x_data)}

    # =========================
    # K线
    # pyecharts格式：[open, close, low, high]
    # =========================
    y_kline = [
        [float(r["open"]), float(r["close"]), float(r["low"]), float(r["high"])]
        for _, r in df_daily.iterrows()
    ]

    # =========================
    # MA
    # =========================
    ma_line = df_daily["ma"].astype(float).tolist() if "ma" in df_daily.columns else None

    # =========================
    # 网格线（多 / 空 / 强平）
    # =========================
    grid_lines = {}
    # 多头网格
    for level in range(1, max_grid + 2):
        col = f"grid_buy_{level}"
        if col in df_daily.columns:
            grid_lines[col] = df_daily[col].astype(float).tolist()
    # 空头网格
    for level in range(1, max_grid + 2):
        col = f"grid_sell_{level}"
        if col in df_daily.columns:
            grid_lines[col] = df_daily[col].astype(float).tolist()
    # 新框架强平线（独立列名）
    for col in ("grid_long_stop", "grid_short_stop"):
        if col in df_daily.columns:
            grid_lines[col] = df_daily[col].astype(float).tolist()

    # =========================
    # 交易点（日线映射）
    # =========================
    buy_pts, sell_pts, force_pts = [], [], []        # 多头开 / 多头平 / 强平
    short_open_pts, short_close_pts = [], []         # 空头开 / 空头平

    for _, r in df_trades.iterrows():
        t = pd.Timestamp(r["trade_date"]).strftime("%Y-%m-%d")
        if t not in ts_index:
            continue

        action = str(r["action"])
        side = int(r["side"]) if "side" in r.index and pd.notna(r["side"]) else 1
        price = float(r["price"])

        if action in FORCE_CLOSE_ACTIONS:
            force_pts.append([t, price])
        elif action in OPEN_ACTIONS or (action == "OPEN" and side > 0):
            if side < 0:
                short_open_pts.append([t, price])
            else:
                buy_pts.append([t, price])
        elif action in CLOSE_ACTIONS or (action == "CLOSE" and side > 0):
            if side < 0:
                short_close_pts.append([t, price])
            else:
                sell_pts.append([t, price])
        elif action in SHORT_OPEN_ACTIONS:
            short_open_pts.append([t, price])
        elif action in SHORT_CLOSE_ACTIONS:
            short_close_pts.append([t, price])

    # =========================
    # 副图：实时净值 / 每日资金占用
    #   实时净值    = total_asset - init_principle  (含未平仓的累计损益, 列: real_nav)
    #   每日资金占用  = daily_capital_used
    # =========================
    sub_chart = None
    if df_summary is not None and not df_summary.empty:
        def _build_day_map(col):
            return {
                pd.Timestamp(r["trade_date"]).strftime("%Y-%m-%d"): r[col]
                for _, r in df_summary.iterrows()
                if col in df_summary.columns
            }

        # ---- 实时净值 ----
        if "real_nav" in df_summary.columns:
            nav_map = _build_day_map("real_nav")
            nav_line = [nav_map.get(t, None) for t in x_data]
        elif "total_asset" in df_summary.columns:
            # 默认取第一天的总资产，若能读到 result_summary.csv 则取配置的初始资金
            init_val = df_summary["total_asset"].iloc[0]
            if result_summary_csv and Path(result_summary_csv).exists():
                try:
                    df_rs = pd.read_csv(result_summary_csv)
                    if "ts_code" in df_rs.columns and code in df_rs["ts_code"].astype(str).values:
                        init_val = float(df_rs[df_rs["ts_code"].astype(str) == code].iloc[0]["init_principle"])
                    elif "init_principle" in df_rs.columns and len(df_rs) > 0:
                        init_val = float(df_rs.iloc[0]["init_principle"])
                except Exception:
                    pass
                    
            nav_map = {
                pd.Timestamp(r["trade_date"]).strftime("%Y-%m-%d"): r["total_asset"] - init_val
                for _, r in df_summary.iterrows()
            }
            nav_line = [nav_map.get(t, None) for t in x_data]
        else:
            nav_line = [None] * len(x_data)

        # ---- 每日资金占用（保留原有，供参考） ----
        if "daily_capital_used" in df_summary.columns:
            cap_map = _build_day_map("daily_capital_used")
            cap_line = [cap_map.get(t, None) for t in x_data]
        else:
            cap_line = [None] * len(x_data)

        sub_chart = (
            Line()
            .add_xaxis(x_data)
            # ① 实时净值 = total P&L including unrealized（蓝色实线）
            .add_yaxis(
                "实时净值",
                nav_line,
                yaxis_index=0,
                is_connect_nones=True,
                symbol="none",
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=2, color="#1a6fbf"),
                itemstyle_opts=opts.ItemStyleOpts(color="#1a6fbf"),
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(y=0, name="零轴")],
                    linestyle_opts=opts.LineStyleOpts(
                        color="#888888", type_="dashed", width=1
                    ),
                    label_opts=opts.LabelOpts(is_show=False),
                ),
            )
            # ② 每日资金占用（灰色细线，次要参考）
            .add_yaxis(
                "每日资金占用",
                cap_line,
                yaxis_index=1,
                is_connect_nones=True,
                symbol="none",
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=1, color="#bbbbbb"),
                itemstyle_opts=opts.ItemStyleOpts(color="#bbbbbb"),
            )
            .extend_axis(
                yaxis=opts.AxisOpts(
                    type_="value",
                    position="right",
                    grid_index=1,
                    is_scale=True,
                    axisline_opts=opts.AxisLineOpts(
                        linestyle_opts=opts.LineStyleOpts(color="#bbbbbb")
                    ),
                    axislabel_opts=opts.LabelOpts(color="#bbbbbb", font_size=10),
                )
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category", grid_index=1, is_scale=True
                ),
                yaxis_opts=opts.AxisOpts(is_scale=True, grid_index=1),
                legend_opts=opts.LegendOpts(
                    is_show=True, pos_top="76%", pos_left="center"
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross",
                    background_color="rgba(245,245,245,0.9)",
                    border_width=1,
                    border_color="#ccc",
                    textstyle_opts=opts.TextStyleOpts(color="#000"),
                ),
            )
        )

    if sub_chart is None:
        sub_chart = (
            Line()
            .add_xaxis(x_data)
            .add_yaxis("占位", [None] * len(x_data),
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(type_="category", grid_index=1, is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True, grid_index=1),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

    # =========================
    # 主图
    # =========================
    kline = (
        Kline()
        .add_xaxis(x_data)
        .add_yaxis(
            "K线", y_kline,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a",
                color0="#14b143",
                border_color="#ef232a",
                border_color0="#14b143"
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{code} 动态网格日线回测"),
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True,
                    areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1]),
                opts.DataZoomOpts(type_="slider", xaxis_index=[0, 1], pos_top="92%"),
            ],
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245,245,245,0.9)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
            legend_opts=opts.LegendOpts(pos_top="2%"),
        )
    )

    # MA
    if ma_line is not None:
        ma_chart = (
            Line()
            .add_xaxis(x_data)
            .add_yaxis(
                "MA",
                ma_line,
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=2, color="#1f77b4"),
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        kline = kline.overlap(ma_chart)

    # 网格线
    grid_colors = ["#f5a623", "#7ed321", "#9013fe", "#50e3c2", "#ff7f50", "#8b0000",
                   "#1abc9c", "#e67e22", "#34495e"]

    stop_cols = {f"grid_buy_{max_grid + 1}", "grid_long_stop", "grid_short_stop"}

    for i, (col, vals) in enumerate(grid_lines.items()):
        color = grid_colors[i % len(grid_colors)]
        is_stop = col in stop_cols
        if col == "grid_long_stop":
            line_name = "多头强平线"
        elif col == "grid_short_stop":
            line_name = "空头强平线"
        elif col == f"grid_buy_{max_grid + 1}":
            line_name = "强平线"
        else:
            line_name = col

        grid_chart = (
            Line()
            .add_xaxis(x_data)
            .add_yaxis(
                line_name,
                vals,
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(
                    width=2.5 if is_stop else 1.5,
                    color=color,
                    type_="solid" if is_stop else "dashed"
                ),
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        kline = kline.overlap(grid_chart)

    # 买点
    if buy_pts:
        sc_buy = (
            Scatter()
            .add_xaxis([p[0] for p in buy_pts])
            .add_yaxis(
                "买点",
                [p[1] for p in buy_pts],
                symbol="triangle",
                symbol_size=14,
                itemstyle_opts=opts.ItemStyleOpts(color="#d0021b"),
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        kline = kline.overlap(sc_buy)

    # 卖点
    if sell_pts:
        sc_sell = (
            Scatter()
            .add_xaxis([p[0] for p in sell_pts])
            .add_yaxis(
                "卖点",
                [p[1] for p in sell_pts],
                symbol="pin",
                symbol_size=14,
                itemstyle_opts=opts.ItemStyleOpts(color="#0a7d2c"),
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        kline = kline.overlap(sc_sell)

    # 空头开
    if short_open_pts:
        sc_so = (
            Scatter()
            .add_xaxis([p[0] for p in short_open_pts])
            .add_yaxis(
                "空头开",
                [p[1] for p in short_open_pts],
                symbol="triangle",
                symbol_rotate=180,
                symbol_size=14,
                itemstyle_opts=opts.ItemStyleOpts(color="#1f6feb"),
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        kline = kline.overlap(sc_so)

    # 空头平
    if short_close_pts:
        sc_sc = (
            Scatter()
            .add_xaxis([p[0] for p in short_close_pts])
            .add_yaxis(
                "空头平",
                [p[1] for p in short_close_pts],
                symbol="rect",
                symbol_size=12,
                itemstyle_opts=opts.ItemStyleOpts(color="#6f42c1"),
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        kline = kline.overlap(sc_sc)

    # 强平点
    if force_pts:
        sc_force = (
            Scatter()
            .add_xaxis([p[0] for p in force_pts])
            .add_yaxis(
                "强平点",
                [p[1] for p in force_pts],
                symbol="diamond",
                symbol_size=16,
                itemstyle_opts=opts.ItemStyleOpts(color="#8b0000"),
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        kline = kline.overlap(sc_force)

    # =========================
    # Grid 输出
    # =========================
    grid = (
        Grid(init_opts=opts.InitOpts(width="100%", height="100vh"))
        .add(kline, grid_opts=opts.GridOpts(pos_left="3%", pos_right="3%", pos_top="5%", height="60%"))
        .add(sub_chart, grid_opts=opts.GridOpts(pos_left="3%", pos_right="3%", pos_top="75%", height="20%"))
    )

    out_path = Path("output") / f"daily_grid_chart_{code}.html"
    out_path.parent.mkdir(exist_ok=True)
    grid.render(str(out_path))

    # ---- 注入右侧量化指标面板 ----
    sidebar_html = _build_sidebar_html(
        code=code,
        result_summary_csv=result_summary_csv,
        holding_stats_csv=holding_stats_csv,
    )
    _inject_sidebar(out_path, sidebar_html)

    print(f"Chart rendered at: {out_path.absolute()}")

    try:
        webbrowser.open(out_path.absolute().as_uri())
    except Exception:
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default=DEFAULT_RUN_CONFIG["code"], help="Code, e.g. sh600019")
    parser.add_argument("--daily_csv", default=DEFAULT_RUN_CONFIG["daily_csv"], help="Daily bars with MA/grid columns")
    parser.add_argument("--trades_csv", default=DEFAULT_RUN_CONFIG["trades_csv"], help="Trade records csv")
    parser.add_argument("--summary_csv", default=DEFAULT_RUN_CONFIG["summary_csv"], help="Daily summary csv")
    parser.add_argument("--holding_stats_csv", default=DEFAULT_RUN_CONFIG["holding_stats_csv"],
                        help="Per-level holding stats csv (for sidebar)")
    parser.add_argument("--result_summary_csv", default=DEFAULT_RUN_CONFIG["result_summary_csv"],
                        help="Backtest result summary csv (for sidebar)")
    parser.add_argument("--max_grid", type=int, default=DEFAULT_RUN_CONFIG["max_grid"])
    return parser


def main() -> None:
    # 无参数时按默认配置运行，便于 IDE 直接点击 Run
    if len(sys.argv) == 1:
        args = argparse.Namespace(**DEFAULT_RUN_CONFIG)
    else:
        args = _build_arg_parser().parse_args()

    daily_path = Path(args.daily_csv)
    trades_path = Path(args.trades_csv)
    summary_path = Path(args.summary_csv) if args.summary_csv else None
    holding_stats_path = Path(args.holding_stats_csv) if getattr(args, "holding_stats_csv", None) else None
    result_summary_path = Path(args.result_summary_csv) if getattr(args, "result_summary_csv", None) else None

    if not daily_path.exists() or not trades_path.exists():
        raise FileNotFoundError(
            "未找到输入文件，请修改 DEFAULT_RUN_CONFIG 或使用命令行参数传入："
            f"\ndaily_csv={daily_path}"
            f"\ntrades_csv={trades_path}"
        )

    if summary_path is not None and not summary_path.exists():
        print(f"warning: summary_csv not found, skip summary subplot: {summary_path}")
        summary_path = None
    if holding_stats_path is not None and not holding_stats_path.exists():
        print(f"warning: holding_stats_csv not found, skip sidebar holding panel: {holding_stats_path}")
        holding_stats_path = None
    if result_summary_path is not None and not result_summary_path.exists():
        print(f"warning: result_summary_csv not found, skip sidebar summary panel: {result_summary_path}")
        result_summary_path = None

    render_daily_grid_chart(
        code=args.code,
        daily_csv=str(daily_path),
        trades_csv=str(trades_path),
        summary_csv=str(summary_path) if summary_path is not None else None,
        max_grid=args.max_grid,
        holding_stats_csv=str(holding_stats_path) if holding_stats_path is not None else None,
        result_summary_csv=str(result_summary_path) if result_summary_path is not None else None,
    )


if __name__ == "__main__":
    main()

