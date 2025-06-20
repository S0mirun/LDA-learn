
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from utils.ship_geometry import *
from utils.time_series_figure  \
    import TimeSeries, make_traj_fig, make_ts_fig, make_traj_and_velo_fig
from utils.visualization import *
from utils.kml import kml_based_txt_to_csv


DIR = os.path.dirname(__file__)
RAW_TS_DIR = f"{DIR}/../raw_data/888-送付データ/2-3_一次解析/"
TS_INFO_TABLE_PATH = f"{DIR}/../raw_data/ts_info_table.xlsx"
TS_HEADER = [
    "time (JST)", "latitude [deg]", "longitude [deg]",
    "GPS quality", "SOG [knot]", "COG [deg]",
    "psi (raw) [deg]", "psi [deg]", "r [deg/m]",
]
#
BLANK_IDS = [15, 16,]


def preprocess():
    #
    ts_info_df = pd.read_excel(TS_INFO_TABLE_PATH, header=0)
    ts_indices = ts_info_df["No."].values.tolist()
    #
    for dir in glob.glob(f"{RAW_TS_DIR}*/"):
        #
        folder_name = os.path.basename(os.path.dirname(dir))
        name_head = folder_name[:2]
        if name_head[-1] == "_":
            folder_id = int(name_head[0])
        else:
            folder_id = int(name_head)
        if folder_id in ts_indices:
            #
            raw_ts_name = folder_name
            ts_id = folder_id
            ts_id_str = str(ts_id).zfill(3)
            #
            ts_info_series = ts_info_df[ts_info_df["No."]==ts_id]
            L = ts_info_series["L"].values[0]
            B = ts_info_series["B"].values[0]
            berthing_flag = True if "着" in ts_info_series["着/離"].values[0] else False
            unberthing_flag = True if "離" in ts_info_series["着/離"].values[0] else False
            ts_info = {
                "No.": ts_id,
                "L": L,
                "B": B,
                "berthing_flag": berthing_flag,
                "unberthing_flag": unberthing_flag,
            }
            #
            for path in glob.glob(f"{RAW_TS_DIR}{raw_ts_name}/1-運動/*.xyz"):
                xyz_path = path
            with open(xyz_path, encoding="shift-jis",) as f:
                ls = f.readlines()
                if not "," in ls[2]:
                    raw_df = pd.read_csv(
                        xyz_path, skiprows=[0,1,], names=TS_HEADER,
                        sep="\s+", encoding="shift-jis",
                    )
                else:
                    raw_df = pd.read_csv(xyz_path, skiprows=[0,1,], names=TS_HEADER,)
            #
            df = prepare_df(raw_df, ts_info)
            log_dir = f"{DIR}/ts_data/original/"
            os.makedirs(f"{log_dir}csv/", exist_ok=True)
            df.to_csv(f"{log_dir}/csv/{ts_id_str}.csv")
            #
            ts = TimeSeries(
                df=df,
                label=ts_id_str, L=L, B=B,
                color=Colors.black, line_style=(0, (1, 0)),
                dt=1.0,
            )
            make_traj_fig(
                ts_list=[ts],
                ship_plot_step_period=100, alpha_ship_shape=0.5,
                fig_size=(5, 5), legend_flag=True,
            )
            save_fig(f"{log_dir}/fig/traj/", f"{ts_id_str}_traj",)
            make_ts_fig(ts_list=[ts], fig_size=(10, 5,))
            save_fig(f"{log_dir}/fig/state/", f"{ts_id_str}_state",)
            make_traj_and_velo_fig(
                ts_list=[ts],
                ship_plot_step_period=100, alpha_ship_shape=0.5,
                fig_size=(14, 7)
            )
            save_fig(f"{log_dir}/fig/traj_and_velo/", f"{ts_id_str}_traj_and_velo",)

def prepare_df(raw_df, ts_info,):
    #
    df = raw_df.copy()
    #
    smoothen_latlon(df)
    #
    time_arr = np.empty(len(df))
    p_x_arr = np.empty(len(df))
    p_y_arr = np.empty(len(df))
    time_origin = JST_str_to_float(df.iloc[0, df.columns.get_loc("time (JST)")])
    lat_origin = df.iloc[-1, df.columns.get_loc("latitude [deg]")]
    lon_origin = df.iloc[-1, df.columns.get_loc("longitude [deg]")]
    angle_from_north = 0.0
    for i in range(len(df)):
        #
        time_arr[i] = JST_str_to_float(df.iloc[i, df.columns.get_loc("time (JST)")]) - time_origin
        #
        p_x_temp, p_y_temp = convert_to_xy(
            df.iloc[i, df.columns.get_loc("latitude [deg]")],
            df.iloc[i, df.columns.get_loc("longitude [deg]")],
            lat_origin, lon_origin, angle_from_north
        )
        p_x_arr[i] = p_x_temp
        p_y_arr[i] = p_y_temp
    #
    df["t [s]"] = time_arr
    df["p_x [m]"] = p_x_arr
    df["p_y [m]"] = p_y_arr
    #
    df["COG [rad]"] = np.deg2rad(df["COG [deg]"].values)
    df["psi [rad]"] = np.deg2rad(df["psi [deg]"].values)
    df["r [deg/s]"] = df["r [deg/m]"] / 60.0
    df["r [rad/s]"] = np.deg2rad(df["r [deg/s]"].values)
    #
    df["U [m/s]"] = knot_to_ms(df["SOG [knot]"])
    df["beta [rad]"] = clip_angle(df["psi [rad]"] - df["COG [rad]"])
    df["beta [deg]"] = np.rad2deg(df["beta [rad]"].values)    
    df["u [m/s]"] = df["U [m/s]"].values * np.cos(df["beta [rad]"].values)
    df["vm [m/s]"] = -df["U [m/s]"].values * np.sin(df["beta [rad]"].values)
    #
    smoothen_ts_1D(df, "t [s]", 1e2)
    # truncate incorrect COG log
    if ts_info["berthing_flag"]:
        for i in range(len(df)):
            COG_inst = df.iloc[i, df.columns.get_loc("COG [deg]")]
            if COG_inst == 360:
                df = df.iloc[:i]
                break
    if ts_info["unberthing_flag"]:
        for i in range(len(df)):
            ts_ind = -(i+1)
            COG_inst = df.iloc[ts_ind, df.columns.get_loc("COG [deg]")]
            if COG_inst == 360.0:
                df = df.iloc[ts_ind:]
                break
    #
    return df

def smoothen_ts_1D(df, label, thr):
    ts_diffs = df[label].diff()
    for i in range(len(df)-1):
        diff = ts_diffs.iloc[i+1]
        if np.abs(diff) > thr:
            smoothen = 0.5 * (
                df.iloc[i, df.columns.get_loc(label)]
                + df.iloc[i+2, df.columns.get_loc(label)]
            )
            df.iloc[i+1, df.columns.get_loc(label)] = smoothen

def smoothen_latlon(df):
    lat_diffs = df["latitude [deg]"].diff()
    lon_diffs = df["longitude [deg]"].diff()
    for i in range(len(df)-1):
        lat_diff = lat_diffs.iloc[i+1]
        lon_diff = lon_diffs.iloc[i+1]
        if np.abs(lat_diff) > 1.0:
            lat_smoothen = 0.5 * (
                df.iloc[i, df.columns.get_loc("latitude [deg]")]
                + df.iloc[i+2, df.columns.get_loc("latitude [deg]")]
            )
            df.iloc[i+1, df.columns.get_loc("latitude [deg]")] = lat_smoothen
        if np.abs(lon_diff) > 1.0:
            lon_smoothen = 0.5 * (
                df.iloc[i, df.columns.get_loc("longitude [deg]")]
                + df.iloc[i+2, df.columns.get_loc("longitude [deg]")]
            )
            df.iloc[i+1, df.columns.get_loc("longitude [deg]")] = lon_smoothen

def JST_str_to_float(str):
    l = str.split(":")
    t = float(l[0]) * 3600.0 + float(l[1]) * 60.0 + float(l[2])
    return t

def make_hist(label):
    #
    ts_info_df = pd.read_excel(TS_INFO_TABLE_PATH, header=0)
    for bid in BLANK_IDS:
        ts_info_df.drop(ts_info_df.index[ts_info_df["No."]==bid], inplace=True,)
    data_ = ts_info_df[label].values
    data = [d for d in data_ if d != "L"]
    #
    set_rcParams()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.hist(
        data,
        bins=10,
        range=(100.0, 350.0),
        color=Colors.black,
        alpha=0.5,
        density=False,
    )
    ax.set_xlabel(r"$L$ [m]")
    ax.set_ylabel(r"Frequency")
    fig.align_labels()
    fig.tight_layout()
    #
    save_fig(
        dir=f"{DIR}/ts_data/original/hist/",
        name=f"hist_{label}",
    )

def make_strip_fig():
    #
    ts_info_df = pd.read_excel(TS_INFO_TABLE_PATH, header=0)
    for bid in BLANK_IDS:
        ts_info_df.drop(ts_info_df.index[ts_info_df["No."]==bid], inplace=True,)
    data_ = ts_info_df["L"].values
    data = [d for d in data_ if d != "L"]
    #
    set_rcParams()
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    sns.stripplot(x=data, color=Colors.black,)
    ax.set_xlabel(r"$L$ [m]")
    fig.align_labels()
    fig.tight_layout()
    #
    save_fig(
        dir=f"{DIR}/ts_data/original/strip/",
        name="strip_L",
    )

def make_fig_Ise_Bay():
    #
    LAT_ORIGIN = 34.45311308979854
    LON_ORIGIN = 136.8840877987753
    ANGLE_FROM_NORTH = 0.0
    #
    kml_based_txt_to_csv(
        txt_path=f"{DIR}/../raw_data/topography/Ise_Bay.txt",
        log_dir=f"{DIR}/../raw_data/topography/",
        csv_name="Ise_Bay",
    )
    df_tpgrph = pd.read_csv(f"{DIR}/../raw_data/topography/Ise_Bay.csv", index_col=0)
    p_x_arrtpgrph = np.empty(len(df_tpgrph))
    p_y_arrtpgrph = np.empty(len(df_tpgrph))
    for i in range(len(df_tpgrph)):
        #
        p_y_temp, p_x_temp = convert_to_xy(
            df_tpgrph.iloc[i, df_tpgrph.columns.get_loc("latitude")],
            df_tpgrph.iloc[i, df_tpgrph.columns.get_loc("longitude")],
            LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
        )
        p_x_arrtpgrph[i] = p_x_temp
        p_y_arrtpgrph[i] = p_y_temp
    #
    set_rcParams()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1, 1, 1)
    # ax setting
    ax.set_xlim(-30000, 45000)
    ax.set_ylim(23000, 75000)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # axins = zoomed_inset_axes(ax, zoom=2.5, loc='upper right')
    axins = ax.inset_axes([0.53, 0.5, 0.45, 0.45])
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec=Colors.black, lw=0.5, ls='dashed')
    axins.set_xlim(-10000, 3000)
    axins.set_ylim(60000, 68000)
    axins.set_aspect('equal')
    axins.set_xticks([])
    axins.set_xticklabels([])
    axins.set_yticks([])
    axins.set_yticklabels([])
    axins.set_aspect('equal')
    # plot topography
    ax.add_patch(
        plt.Polygon(
            np.transpose(np.array([
                p_x_arrtpgrph,
                p_y_arrtpgrph,
            ])),
            fill=True, alpha=0.5,
            color=Colors.black, linewidth=0,
            # label = "Topography",
        )
    )
    axins.add_patch(
        plt.Polygon(
            np.transpose(np.array([
                p_x_arrtpgrph,
                p_y_arrtpgrph,
            ])),
            fill=True, alpha=0.5,
            color=Colors.black, linewidth=0,
            # label = "Topography",
        )
    )
    # plot scale
    scalebar_length = 10000
    x0, y0 = -25000, 25000  # left and right edges of scale bar
    tick_interval = 1000
    lw_m = 2.5
    ax.plot([x0, x0 + scalebar_length], [y0, y0], color=Colors.black, linewidth=lw_m)
    num_ticks = scalebar_length // tick_interval + 1
    for i in range(num_ticks):
        xtick = x0 + i * tick_interval
        ax.plot(
            [xtick, xtick],
            [y0, y0 + 1000],
            color=Colors.black,
            linewidth=lw_m if i==0 or i==int(num_ticks / 2) or i==num_ticks-1 else 1,
        )
        if i == 0:
            ax.text(xtick, y0 + 3000, "0", ha='center', va='top',)
        if i == num_ticks - 1:
            ax.text(xtick, y0 + 3000, f'{scalebar_length} m' , ha='center', va='top',)

    # plot path
    for i_path, path in enumerate(glob.glob(f"{DIR}/ts_data/original/csv/*.csv")):
        #
        df = pd.read_csv(path, index_col=0)
        p_x_arr = np.empty(len(df))
        p_y_arr = np.empty(len(df))
        for i in range(len(df)):
            p_y_temp, p_x_temp = convert_to_xy(
                df.iloc[i, df.columns.get_loc("latitude [deg]")],
                df.iloc[i, df.columns.get_loc("longitude [deg]")],
                LAT_ORIGIN, LON_ORIGIN, ANGLE_FROM_NORTH
            )
            p_x_arr[i] = p_x_temp
            p_y_arr[i] = p_y_temp
        #
        ax.plot(
            p_x_arr,
            p_y_arr,
            c=Colors.black,
            linewidth=0.5,
            label="Path" if i_path==0 else None,
        )
        axins.plot(
            p_x_arr,
            p_y_arr,
            c=Colors.black,
            linewidth=1.0,
        )
    #
    ax.legend(
        bbox_to_anchor=(0, 1),
        loc='upper left',
        fontsize=15
    )
    #
    fig.align_labels()
    fig.tight_layout()
    #
    save_fig(
        dir=f"{DIR}/ts_data/original/fig/Ise_Bay/",
        name="all_paths"
    )


if __name__ == "__main__":
    preprocess()
    make_hist("L")
    make_strip_fig()
    make_fig_Ise_Bay()
    #
    print("\nDone\n")
