"""

"""
import numpy as np
from textwrap import fill

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", 200)

import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style("white")

import scipy
import scipy.cluster.hierarchy as sch
import statsmodels.api as sm

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook"
pio.templates.default = "simple_white"

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

config = {
    "toImageButtonOptions": {
        "format": "svg",  # one of png, svg, jpeg, webp
        # 'filename': 'custom_image',
        "height": None,
        "width": None,
        # 'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
    }
}

translation = {
    "GENDER_NONBINARY_CUS": "Gender",
    "QUOTAGERANGE": "Age Range",
    "Q3_Q7_1": "Work w/ External",
    "Q3_Q7_2": "Works on Behavior",
    "Q39_continent": "Continent of Residence",
    "Home work": "Works in Residence Country",
    "Q41": "Education Level",
    "Q43": "Work Sector",
    "Q45": "Professional Engagement",
    "Q45_simple": "Professional Engagement Simplified",
    "Q46": "Seniority",
    "Q47": "Work Context",
    "Q10_Q15_3": "Positive Emotions",
    "Q10_Q15_4": "Negative Emotions",
    "Q10_Q15_5": "Personalization",
    "Q10_Q15_6": "Observability",
    "Q10_Q15_7": "Social Norms",
    "Q10_Q15_8": "Public Commitments",
    "Q18_Q25_1": "Attention",
    "Q18_Q25_2": "Reminders",
    "Q18_Q25_3": "Planning",
    "Q18_Q25_4": "Information",
    "Q18_Q25_5": "Rules and Regulation",
    "Q18_Q25_6": "Material Incentives",
    "Social Influences": "Any Social Influence",
    "Choice Architecture": "Any Choice Architecture",
    "Emotional Appeals": "Any Emotional Appeals",
    "Q3_Q7_3": "Defined Target",
    "Q3_Q7_4": "Defined Behaviors",
    "Q3_Q7_5": "Systems Map",
    "Q3_Q7_6": "Quantitative Research",
    "Q3_Q7_7": "Qualitative Research",
    "Frame": "Any Frame Step",
    "Empathize": "Any Empathize Step",
    "Q10_Q15_2": "Map (Data for Desigh)",
    "Q10_Q15_1": "Ideate (BeSci for Design)",
    "Q18_Q25_7": "Pilot",
    "Q18_Q25_8": "Evaluate",
    "Experimental": "RCT",
    "Quasi-Experimental": "Diff-in-Diff or Matched",
    "Q27": "Durability",
    "Causal Estimation": "Causal Inference",
    "Q29_ranked": "BCD definition groups",
    "Q29_dontKnow": "Did not know how to define BCD",
    "Q30": "BCD applied to work",
    "Q31": "BCD boosts effectiveness",
    "Q31_factorized": "BCD boosts effectiveness (3 = neutral)",
    "Q32": "Talked about BCD",
    "Q33": "BCD Training",
    "SCB_members": "Is SCB member (otherwise IUCN)",
    "Q42_1": "Education: Biological sciences",
    "Q42_2": "Education: Social sciences",
    "Q42_3": "Education: Interdisciplinary",
    "Q42_4": "Education: Non-biological sciences",
    "Q42_5": "Education: Humanities",
    "Q42_6": "Education: Behavioral sciences",
    "Q42_7": "Education: Business, Management, or Law",
    "Q42_8": "Education: Communications or Marketing",
    "Q42_9": "Education: None of the above",
    "Q35_1": "Source: Peer-reviewed journal articles",
    "Q35_2": "Source: IUCN specialist groups",
    "Q35_3": "Source: Blogs or stories by practitioners",
    "Q35_4": "Source: List-serves or google groups",
    "Q35_5": "Source: Colleagues",
    "Q35_6": "Source: Conferences, workshops, other events",
    "Q35_7": "Source: Others",
    "Q35_71": "Trusted Source: Specified",
    "Q44_1": "Field: Business and biodiversity",
    "Q44_2": "Field: Climate Change",
    "Q44_3": "Field: Ecosystem Management",
    "Q44_4": "Field: Environmental Law",
    "Q44_5": "Field: Forests",
    "Q44_6": "Field: Gender",
    "Q44_7": "Field: Global Policy",
    "Q44_8": "Field: Governance and Rights",
    "Q44_9": "Field: Marine and Polar",
    "Q44_10": "Field: Nature-based solutions",
    "Q44_11": "Field: Protected Areas",
    "Q44_12": "Field: Science and Economics",
    "Q44_13": "Field: Species",
    "Q44_14": "Field: Water",
    "Q44_15": "Field: World Heritage",
    "Q44_16": "Field: Not applicable",
    "Q44_17": "Field: Other",
    "Q44_171": "Field Specified",
    "BCD_score": "# of BCD steps realized",
    "Lever_score": "Interventions leaning towards BeSci",
    "BCD_accomplished": "# of BCD steps realized (in order)",
    "Q29_CODED_CB_1": "BCD def: Plan designed for interventions that influence/change peoples`s behavior",
    "Q29_CODED_CB_2": "BCD def: The way people relates to the environment has an impact on nature",
    "Q29_CODED_CB_3": "BCD def: Change of attitude towards the environment conservation",
    "Q29_CODED_CB_4": "BCD def: Measurable behavior change",
    "Q29_CODED_CB_5": "BCD def: Strategies/activity based on a focus on human behavior",
    "Q29_CODED_CB_6": "BCD def: Change /abandon former practices/attitudes and adopting new ones",
    "Q29_CODED_CB_7": "BCD def: Not familiar with the concept",
    "Q29_CODED_CB_8": "BCD def: Effective action against existing problems",
    "Q29_CODED_CB_9": "BCD def: Social science/engineering",
    "Q29_CODED_CB_10": "BCD def: People participation/collaboration",
    "Q29_CODED_CB_11": "BCD def: New approach to behavior change analysis",
    "Q29_CODED_CB_12": "BCD def: Knowledge about behavior to design environmental solutions",
    "Q29_CODED_CB_13": "BCD def: Design principles that use psychology/marketing to encourage behavior change",
    "Q29_CODED_CB_14": "BCD def: Resilience",
    "Q29_CODED_CB_15": "BCD def: Mutual understanding/communication",
    "Q29_CODED_CB_97": "BCD def: Nothing/none",
    "Q29_CODED_CB_98": "BCD def: Other",
    "Q29_CODED_CB_99": "BCD def: Don`t know",
    "Levers_total": "Number of different levers used",
    "Q29": "If you had to define it, what does behavior-centered design mean to you?",
    "Q34": "When you think of behavior-centered design, what organizations comes to mind?",
    "Q35": "What sources of information do you trust about new tools and methodologies in the conservation or environment fields?",
    "Q40": "In what countries do you currently work?",
    "Q42": "Which of the following best describes your educational specialization?",
    "Q44": "What topic areas do you work on?",
}


def translate(question):
    try:
        return translation[question]
    except:
        return question


def translate_list(question):
    array = []
    for ll in question:
        try:
            array.append(translation[ll])
        except:
            array.append(ll)

    return array
    # return translation[question]


# ==========================================================================================


def plot_MultipleChoice(
    questionNo,
    df,
    meta,
    iv=False,
    plot=False,
    barmode="group",
    orientation="v",
    limit=None,
):
    """
    Generate count of answers and combines Multiple Choice Answers into singular column of lists
    """
    questionNo_ = questionNo + "_"

    meta_index = [i for i in meta.T.columns if questionNo_ in i]
    meta_index = np.array(meta_index)[
        [False if x == "Q44_171" else True for x in meta_index]
    ]
    meta_index = np.array(meta_index)[
        [False if x == "Q35_71" else True for x in meta_index]
    ]

    definitions = []
    for ind in meta_index:
        colname = meta.T[ind].values[0]
        start = colname.find("(")
        end = colname.find(")")
        definitions.extend([colname[start + 1 : end]])
    definitions = np.array(definitions)

    qq = [i for i in df.columns if questionNo_ in i]

    # plot grouped by a specific IV
    if iv:
        i = 0
        definitions_df = {}
        for define, question in zip(definitions, qq):
            for iv_col in df[iv].unique():
                subdf = df.loc[df[iv] == iv_col]
                definitions_df[i] = [
                    define,
                    iv_col,
                    sum(subdf[question].values == "Yes"),
                ]
                i += 1
        definitions_df = pd.DataFrame(definitions_df).T.rename(
            columns={0: translate(questionNo), 1: translate(iv), 2: "Count"}
        )

        # Set appropriate colors

        # check if the data is categorical + ordered
        colour = sb.diverging_palette(
            240, 10, n=len(definitions_df[translate(iv)].unique())
        ).as_hex()

        if df[iv].dtypes.name == "category":
            if not df[iv].cat.ordered:
                colour = sb.color_palette(
                    "crest", n_colors=len(definitions_df[translate(iv)].unique())
                ).as_hex()

        # create the order of the categories
        reordered = {}
        reordered[translate(questionNo)] = (
            definitions_df.groupby(translate(questionNo))["Count"]
            .sum()
            .sort_values(ascending=False)
            .index
        )
        reordered[translate(iv)] = (
            definitions_df[translate(iv)].sort_values().unique()[::-1]
        )

        if plot:
            fig = px.bar(
                definitions_df.loc[definitions_df["Count"] != 0],
                x=translate(questionNo),
                y="Count",
                color=translate(iv),
                barmode=barmode,
                title=translate(questionNo),
                color_discrete_sequence=colour,
                category_orders=reordered,
            )
            fig.update_layout(
                xaxis_tickangle=45, title={"x": 0.5, "xanchor": "center",},
            )
            fig["layout"]["font"]["size"] = 8
            fig.show()

    else:
        definitions_df = {}
        for define, question in zip(definitions, qq):
            definitions_df[define] = sum(df[question].values == "Yes")
        definitions_df = pd.Series(definitions_df)

        subDF = definitions_df.reset_index(level=0)
        subDF.rename(columns={0: "Count", "index": "Variable"}, inplace=True)
        subDF["% Count"] = subDF["Count"].apply(lambda x: (x / len(df)) * 100)
        subDF = subDF.sort_values("% Count", ascending=False)
        subDF = subDF[subDF["Count"] != 0]

        meta_index = [i for i in meta.T.columns if questionNo in i]
        colname = meta.T[meta_index[0]].values[0]
        start = colname.find("- ")
        definition = colname[start + 1 :]

        # subDF["Variable"] = translate_list(subDF["Variable"].values)

        if limit:
            subDF = subDF.iloc[0:limit]

        if plot:
            if orientation == "v":
                fig = px.bar(
                    subDF,
                    x="Variable",
                    y="% Count",
                    color="Count",
                    title=translate(questionNo),
                    color_continuous_scale="Aggrnyl_r",
                    orientation=orientation,
                )
                fig.update_layout(
                    xaxis_tickangle=45, title={"x": 0.5, "xanchor": "center",},
                )
                # fig.update_traces(texttemplate="%{text:%}", textposition="outside")
                fig.update_layout(height=600, width=70 * len(subDF["Variable"].unique()))
            elif orientation == "h":
                subDF = subDF.sort_values("% Count", ascending=True)
                fig = px.bar(
                    subDF,
                    x="% Count",
                    y="Variable",
                    color="Count",
                    title=translate(questionNo),
                    color_continuous_scale="Aggrnyl_r",
                    orientation=orientation,
                    text=["{0:.1f}%".format(i) for i in subDF["% Count"]],
                )
                fig.update_layout(
                    title={"x": 0.5, "xanchor": "center",}, coloraxis_showscale=False,
                )
                fig.update_traces(texttemplate="%{text}", textposition="auto")
                fig.update_layout(
                    width=800, height=25 * len(subDF["Variable"].unique()) + 200
                )

            fig["layout"]["font"]["size"] = 9
            fig.show()

    return definitions_df, meta_index


def convert(x):
    import pycountry_convert as pc

    return pc.convert_continent_code_to_continent_name(
        pc.country_alpha2_to_continent_code(
            pc.country_name_to_country_alpha2(x, cn_name_format="default")
        )
    )


def combine_questions(df):
    data = []
    for row in df.values:
        if np.any(row == "Yes"):
            data.extend(["Yes"])
        else:
            if np.any(row == "Not sure"):
                data.extend(["Not sure"])
            else:
                data.extend(["No"])
    return pd.Series(
        pd.Categorical(data, categories=["No", "Not sure", "Yes"], ordered=True)
    )

def count_questions(df):
    cols = len(df.columns)

    calc_dict = {"Yes" : 1/cols,
                "Not sure" : 0,
                "No" : 0}

    return df.replace(calc_dict).sum(axis=1)



def factorize_survey(df, drop_first=False, removeUnsure=False, standardize=True):

    from sklearn.preprocessing import StandardScaler

    subDF = df.copy()

    if removeUnsure:
        subDF.replace(to_replace="Yes", value=1, inplace=True)
        subDF.replace(to_replace="No", value=0, inplace=True)
        subDF.replace(to_replace="Not sure", value=np.nan, inplace=True)

    odered_mask = subDF.select_dtypes(include=["category"]).apply(
        lambda x: x.cat.ordered
    )
    odered_mask = subDF.select_dtypes(include=["category"]).columns[odered_mask]

    for col in odered_mask:
        if (col in ["Q41", "Q46", "QUOTAGERANGE", "Q31"]) or (
            len(subDF[col].cat.categories) <= 3
        ):
            subDF[col] = subDF[col].cat.codes
        else:
            subDF[col] = subDF[col].cat.as_unordered()
            if removeUnsure:
                try:
                    subDF[col] = subDF[col].astype("boolean")
                except:
                    continue

    subDF.columns = translate_list(subDF.columns.values)

    unodered_mask = subDF.select_dtypes(include=["category"]).apply(
        lambda x: not x.cat.ordered
    )
    unodered_mask = subDF.select_dtypes(include=["category"]).columns[unodered_mask]
    subDF = pd.get_dummies(subDF, drop_first=drop_first, columns=unodered_mask)

    if standardize:
        cc = subDF.columns
        return pd.DataFrame(StandardScaler().fit_transform(subDF), columns=cc)
    else:
        return subDF


# =========================================================================


def plot_heatmap(df, metric="corr", clustered=False):
    import plotly.graph_objs as go
    import plotly.figure_factory as ff
    import plotly.express as px
    import ppscore as pps
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr
    import phik
    from phik import resources, report

    def pearsonr_pval(x, y):
        return pearsonr(x, y)[1]

    if metric == "pps":
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)

        matrix_df = pps.matrix(df, sorted=clustered)[["x", "y", "ppscore"]].pivot(
            columns="x", index="y", values="ppscore"
        )
        matrix_df = matrix_df.loc[df.columns, df.columns]

        hovertext = list()
        for yi, yy in enumerate(translate_list(matrix_df.columns.values)):
            hovertext.append(list())
            for xi, xx in enumerate(translate_list(matrix_df.index.values)):
                hovertext[-1].append(
                    "X (column): {}<br />Y (row): {}<br />".format(xx, yy)
                )

        fig = ff.create_annotated_heatmap(
            z=matrix_df.round(2).values,
            x=translate_list(matrix_df.index.values),
            y=translate_list(matrix_df.columns.values),
            colorscale=px.colors.sequential.BuPu,
            zmin=0,
            zmax=1,
            hoverinfo="text",
            text=hovertext,
        )

        fig.update_layout(
            title={
                "text": "Ipsos Survey Predictive Power Score",
                "x": 0.5,
                "font": dict(family="Arial", size=12,),
            },
            width=1000,
            height=1000,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange="reversed",
            xaxis=go.layout.XAxis(tickangle=40),
        )

        fig["data"][0]["showscale"] = True
        fig["layout"]["xaxis"].update(side="bottom")
        fig["layout"]["font"]["size"] = 6
        fig.show()

        return matrix_df

    elif metric == "phi":

        corr = df.phik_matrix()
        sig = df.significance_matrix()

        # hovertext = list()
        # for yi, yy in enumerate(translate_list(corr.columns.values)):
        #     hovertext.append(list())
        #     for xi, xx in enumerate(translate_list(corr.index.values)):
        #         hovertext[-1].append(
        #             "X: {}<br />Y: {}<br />𝜙k: {:.3f}".format(
        #                 xx, yy, corr.values[yi][xi]
        #             )
        #         )

        if clustered == False:
            hovertext = list()
            for yi, yy in enumerate(translate_list(corr.columns.values)):
                hovertext.append(list())
                for xi, xx in enumerate(translate_list(corr.index.values)):
                    hovertext[-1].append(
                        "X: {}<br />Y: {}<br />𝜙k: {:.3f} <br />*p*: {:.3f}".format(
                            xx, yy, corr.values[yi][xi], sig.values[yi][xi]
                        )
                    )

            heatmap = go.Heatmap(
                z=corr,  # .mask(mask),
                x=translate_list(corr.columns.values),
                y=translate_list(corr.columns.values),
                colorscale=px.colors.sequential.BuPu,
                text=hovertext,
                zmin=0,
                zmax=1,
            )

            layout = go.Layout(
                title={
                    "text": "Ipsos Survey Correlation Matrix",
                    "x": 0.5,
                    "font": dict(family="Arial", size=12,),
                },
                width=1200,
                height=1000,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                yaxis_autorange="reversed",
                xaxis=go.layout.XAxis(tickangle=40),
            )

            fig = go.Figure(data=[heatmap], layout=layout)
            fig["layout"]["font"]["size"] = 6
            fig.show()

        else:
            # get data
            data_array = corr.values
            labels = corr.columns.values

            # Initialize figure by creating upper dendrogram
            fig = ff.create_dendrogram(
                data_array,
                orientation="bottom",
                color_threshold=2.1,
                labels=labels,
                colorscale=px.colors.qualitative.Antique,
            )
            for i in range(len(fig["data"])):
                fig["data"][i]["yaxis"] = "y2"

            # Create Side Dendrogram
            dendro_side = ff.create_dendrogram(
                data_array,
                orientation="left",
                color_threshold=2.1,
                colorscale=px.colors.qualitative.Antique,
            )
            for i in range(len(dendro_side["data"])):
                dendro_side["data"][i]["xaxis"] = "x2"

            # Add Side Dendrogram Data to Figure
            fig.add_traces(dendro_side["data"])

            # Create Heatmap
            dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
            dendro_leaves = list(map(int, dendro_leaves))
            data_dist = pdist(data_array)
            heat_data = corr.iloc[dendro_leaves, dendro_leaves]  # data_array

            hovertext = list()
            for yi, yy in enumerate(translate_list(heat_data.columns.values)):
                hovertext.append(list())
                for xi, xx in enumerate(translate_list(heat_data.index.values)):
                    hovertext[-1].append(
                        "X: {}<br />Y: {}<br />𝜙k: {:.3f}".format(
                            xx, yy, heat_data.values[yi][xi]
                        )
                    )

            heatmap = [
                go.Heatmap(
                    x=translate_list(heat_data.columns.values),
                    y=translate_list(heat_data.columns.values),
                    z=heat_data,
                    colorscale="BuPu",
                    hoverinfo="text",
                    text=hovertext,
                    zmin=0,
                    zmax=1,
                )
            ]

            heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
            heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

            # Add Heatmap Data to Figure
            for data in heatmap:
                fig.add_trace(data)

            # Edit xaxis
            fig.update_layout(
                xaxis={
                    "domain": [0, 0.85],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "showticklabels": False,
                    "ticks": "",
                }
            )
            # Edit xaxis2
            fig.update_layout(
                xaxis2={
                    "domain": [0.855, 1],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "showticklabels": False,
                    "ticks": "",
                }
            )
            # Edit yaxis
            fig.update_layout(
                yaxis={
                    "domain": [0, 0.85],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "ticks": "",
                    "tickvals": dendro_side["layout"]["yaxis"]["tickvals"],
                    "ticktext": translate_list(heat_data.columns.values),
                }
            )

            # Edit yaxis2
            fig.update_layout(
                yaxis2={
                    "domain": [0.85, 1],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "showticklabels": False,
                    "ticks": "",
                }
            )

            fig["layout"]["font"]["size"] = 6

            fig.update_layout(
                title={
                    "text": "Ipsos Survey Correlation Clustermap",
                    "x": 0.5,
                    "font": dict(family="Arial", size=12,),
                },
                width=1200,
                height=1050,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                # yaxis_autorange='reversed',
                # xaxis_autorange='reversed',
                showlegend=False,
                hovermode="closest",
                margin=dict(pad=10),
            )

            # Plot!
            fig.show()

        return corr

    elif metric == "corr":
        df2 = df.copy()

        # mask = ~np.tril(np.ones_like(corr, dtype=bool))

        df2 = factorize_survey(df2, standardize=False)
        corr = df2.corr()
        p_val = df2.corr(method=pearsonr_pval)

        if clustered == False:
            hovertext = list()
            for yi, yy in enumerate(corr.columns.values):
                hovertext.append(list())
                for xi, xx in enumerate(corr.index.values):
                    hovertext[-1].append(
                        "X: {}<br />Y: {}<br />𝜙k: {:.3f} <br />p: {:.3f}".format(
                            xx, yy, corr.values[yi][xi], p_val.values[yi][xi]
                        )
                    )
            heatmap = go.Heatmap(
                z=corr,  # .mask(mask),
                x=corr.columns.values,
                y=corr.columns.values,
                colorscale=px.colors.diverging.RdBu_r,
                zmin=-1,
                zmax=1,
                text=hovertext,
            )

            layout = go.Layout(
                title={
                    "text": "Ipsos Survey Correlation Matrix",
                    "x": 0.5,
                    "font": dict(family="Arial", size=12,),
                },
                width=1200,
                height=1000,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                yaxis_autorange="reversed",
                xaxis=go.layout.XAxis(tickangle=40),
            )

            fig = go.Figure(data=[heatmap], layout=layout)
            fig["layout"]["font"]["size"] = 6
            fig.show()

        else:
            # get data
            data_array = corr.values
            labels = corr.columns.values

            # Initialize figure by creating upper dendrogram
            fig = ff.create_dendrogram(
                data_array,
                orientation="bottom",
                color_threshold=2.1,
                labels=labels,
                colorscale=px.colors.qualitative.Antique,
            )
            for i in range(len(fig["data"])):
                fig["data"][i]["yaxis"] = "y2"

            # Create Side Dendrogram
            dendro_side = ff.create_dendrogram(
                data_array,
                orientation="left",
                color_threshold=2.1,
                colorscale=px.colors.qualitative.Antique,
            )
            for i in range(len(dendro_side["data"])):
                dendro_side["data"][i]["xaxis"] = "x2"

            # Add Side Dendrogram Data to Figure
            fig.add_traces(dendro_side["data"])

            # Create Heatmap
            dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
            dendro_leaves = list(map(int, dendro_leaves))
            data_dist = pdist(data_array)
            heat_data = corr.iloc[dendro_leaves, dendro_leaves]  # data_array

            hovertext = list()
            for yi, yy in enumerate(heat_data.index.values):
                hovertext.append(list())
                for xi, xx in enumerate(heat_data.columns.values):
                    hovertext[-1].append(
                        "X: {}<br />Y: {}<br />pearson R: {:.3f}".format(
                            xx, yy, heat_data.values[yi][xi]
                        )
                    )

            heatmap = [
                go.Heatmap(
                    x=heat_data.columns.values,
                    y=heat_data.columns.values,
                    z=heat_data,
                    colorscale="RdBu_r",
                    hoverinfo="text",
                    text=hovertext,
                    zmin=-1,
                    zmax=1,
                )
            ]

            heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
            heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

            # Add Heatmap Data to Figure
            for data in heatmap:
                fig.add_trace(data)

            # Edit xaxis
            fig.update_layout(
                xaxis={
                    "domain": [0, 0.85],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "showticklabels": False,
                    "ticks": "",
                }
            )
            # Edit xaxis2
            fig.update_layout(
                xaxis2={
                    "domain": [0.855, 1],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "showticklabels": False,
                    "ticks": "",
                }
            )
            # Edit yaxis
            fig.update_layout(
                yaxis={
                    "domain": [0, 0.85],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "ticks": "",
                    "tickvals": dendro_side["layout"]["yaxis"]["tickvals"],
                    "ticktext": heat_data.columns.values,
                }
            )

            # Edit yaxis2
            fig.update_layout(
                yaxis2={
                    "domain": [0.85, 1],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "showticklabels": False,
                    "ticks": "",
                }
            )

            fig["layout"]["font"]["size"] = 6

            fig.update_layout(
                title={
                    "text": "Ipsos Survey Correlation Clustermap",
                    "x": 0.5,
                    "font": dict(family="Arial", size=12,),
                },
                width=1200,
                height=1050,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                # yaxis_autorange='reversed',
                # xaxis_autorange='reversed',
                showlegend=False,
                hovermode="closest",
                margin=dict(pad=10),
            )

            # Plot!
            fig.show()

        return corr


#%%


def combine_questions_short(df):
    data = []
    for row in df.values:
        if np.any(row == "Yes"):
            data.extend(["Yes"])
        else:
            if np.any(row == "Not sure"):
                data.extend(["Not sure"])
            else:
                data.extend(["No"])
    return pd.Series(pd.Categorical(data, categories=["No", "Yes"], ordered=True))


#%% by


def plot_groupBy(df, dv, iv, barmode="stack", normalize=True):
    """
    """

    # nrows = 1
    ncols = len(dv)
    nrows = 1

    spacing = 0.2
    if ncols != 1:
        if (1 / (ncols - 1)) < 0.3:
            spacing = (1 / (ncols - 1)) / 2
            # spacing = 0.2 / ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=(translate_list(dv)),
        horizontal_spacing=spacing
        # horizontal_spacing=0,
    )

    ii = 1
    a = 1
    for i, index in enumerate(dv):
        subDF = df[[iv, index]].value_counts()
        subDF = subDF.reset_index().sort_values([iv, index], ascending=False)

        colour = sb.color_palette("crest", n_colors=len(subDF[iv].unique())).as_hex()

        if not df[[index]].select_dtypes(include=[np.number]).columns.empty:

            if len(subDF[iv].unique()) <= 3:
                colour = sb.diverging_palette(
                    240, 10, n=len(subDF[iv].unique())
                ).as_hex()

            reordered = {}
            reordered[translate(iv)] = df[iv].sort_values().unique()[::-1]
            reordered[translate(index)] = df[index].sort_values().unique()[::-1]

            trace = px.box(
                df.rename(columns={index: translate(index), iv: translate(iv)}),
                y=translate(index),
                color=translate(iv),
                x=translate(iv),
                # box=True,  # draw box plot inside the violin
                points="all",
                color_discrete_sequence=colour,
                category_orders=reordered,
                # points='all'  # can be 'outliers', or False
            )

            subDF.rename(
                columns={index: translate(index), iv: translate(iv)}, inplace=True
            )

            # width = 350 * ncols
            # width = len(subDF[translate(iv)].unique()) * 22

        else:

            reordered = {}
            reordered[translate(iv)] = df[iv].sort_values().unique()
            # reordered["variable"] = df[iv].sort_values().unique()[::-1]
            reordered[translate(index)] = df[index].sort_values().unique()[::-1]

            if df[index].dtypes.name == "category":
                if df[index].cat.ordered:
                    # reordered[translate(index)] = reordered[translate(index)][::-1]
                    colour = sb.diverging_palette(
                        240, 10, n=len(subDF[index].unique())
                    ).as_hex()

            elif df[index].dtypes.name == "boolean" or df[index].dtypes.name == "bool":
                colour = sb.diverging_palette(
                    240, 10, n=len(subDF[index].unique())
                ).as_hex()

            totals = subDF.groupby(iv)[0].sum()

            # Calculate Error
            if barmode == "stack":
                for ff, row in subDF.iterrows():
                    subDF.loc[ff, "Percentage"] = (
                        (row[0] / totals[row[iv]]) * 100
                    ).round(3)

            elif barmode == "group":
                for ff, row in subDF.iterrows():
                    subDF.loc[ff, "Percentage"] = (
                        (row[0] / totals[row[iv]]) * 100
                    ).round(3)

                for ind in np.unique(subDF[iv].values):
                    counts = subDF[subDF[iv] == ind].loc[:, 0].values
                    subDF.loc[subDF[iv] == ind, ["Error_low", "Error_high"]] = (
                        sm.stats.multinomial_proportions_confint(counts) * 100
                    )
                    subDF.loc[subDF[iv] == ind, ["Error_low", "Error_high"]] = np.abs(
                        subDF.loc[subDF[iv] == ind, ["Error_low", "Error_high"]].values
                        - subDF.loc[subDF[iv] == ind, ["Percentage"]].values
                    )

            if not normalize:
                for ff, row in subDF.iterrows():
                    subDF.loc[ff, ["Percentage", "Error_low", "Error_high"]] = (
                        subDF.loc[ff, ["Percentage", "Error_low", "Error_high"]] / 100
                    ) * totals[row[iv]]

                subDF["Percentage"] = subDF["Percentage"].round(0)

            subDF.rename(
                columns={index: translate(index), iv: translate(iv)}, inplace=True
            )
            # subDF["variable"] = subDF[translate(iv)]

            if barmode == "stack":
                # reordered[translate(iv)] = df[iv].sort_values().unique()
                trace = px.bar(
                    subDF,
                    x=translate(iv),
                    y="Percentage",
                    color=translate(index),
                    barmode=barmode,
                    color_discrete_sequence=colour,
                    category_orders=reordered,
                )
            elif barmode == "group":
                trace = px.bar(
                    subDF,
                    x=translate(iv),
                    y="Percentage",
                    error_y="Error_high",
                    error_y_minus="Error_low",
                    color=translate(index),
                    barmode=barmode,
                    color_discrete_sequence=colour,
                    category_orders=reordered,
                )

            # elif barmode == "star":
            #     for iii in subDF[translate(index)].unique():

            #     trace = px.line_polar(subDF, r='translate(iv)', theta='Percentage', line_close=True)

            # if barmode == "stack":
            #     if len(subDF[translate(iv)].unique()) <= 3:
            #         width = 200 * ncols
            #     else:
            #         width = (len(subDF[translate(iv)].unique()) * 44) * ncols
            # elif barmode == "group":
            #     width = (len(subDF[translate(iv)].unique()) * 100) * ncols
            #     if width < 350:
            #         width = 350

        figure1_traces = []
        for tt in range(len(trace["data"])):
            figure1_traces.append(trace["data"][tt])

        for tt in figure1_traces:
            fig.append_trace(tt, row=a, col=ii)

        ii += 1
        # if ii >= 7:
        #     ii = 1
        #     a += 1

    if barmode == "stack":
        if len(subDF[translate(iv)].unique()) <= 3:
            width = 200 * ncols
            # width = 100 * (np.unique(df.values) * ncols)
        else:
            width = (len(subDF[translate(iv)].unique()) * 44) * ncols
    elif barmode == "group":
        width = (len(subDF[translate(iv)].unique()) * 100) * ncols
        if width < 350:
            width = 350

    fig.update_layout({"barmode": barmode})
    fig["layout"]["font"]["size"] = 8
    fig.update_layout(
        width=width,
        # height=350,
        showlegend=False,
        title={
            "text": translate(iv),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.05,
            "yanchor": "bottom",
            "font": dict(family="Arial", size=10),
        },
    )
    fig.update_annotations(font_size=10)

    fig.show()


# ===================================================================================


#%%
def plot_single_proportions_2(
    df,
    meta_data,
    printQuestions=False,
    title="Univariate Analysis",
    orientation="v",
    barmode="stack",
    showlegend=False,
):
    """
    """

    print("\n")
    numerical = 0
    factors = []

    cat_df = df.select_dtypes(include=["category"])
    if not cat_df.empty:
        factors = np.array(
            [cat_df[column].cat.categories.tolist() for column in cat_df.columns],
            dtype=object,
        )
        factors = [list(x) for x in set(tuple(x) for x in factors)]
        if type(factors[0]) != list:
            factors = np.array([factors])

    if df.select_dtypes(include=["boolean"]).empty:
        indices = []
    else:
        indices = [df.select_dtypes(include=["boolean"]).columns]

    if not df.select_dtypes(include=[np.number]).columns.empty:
        numerical += 1

    if np.size(factors) != 0:
        for cc in factors:
            sub_indices = []
            for ii, column in enumerate(cat_df.columns):
                if np.array_equal(cat_df[column].cat.categories, cc):
                    sub_indices.extend([column])
            indices.append(sub_indices)
    else:
        indices = [[cc] for cc in df.columns]
        numerical = 0

    nrows = 1
    ncols = len(indices) + numerical

    spacing = 0.3
    if ncols != 1:
        if (1 / (ncols - 1)) < 0.3:
            spacing = (1 / (ncols - 1)) / 2
            # spacing = 0.2 / ncols

    fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=spacing,)
    max_answers = 1
    for ii, index in enumerate(indices):

        reordered = {}
        # reordered["Question"] = df[index[0]].sort_values().unique()[::-1]
        reordered["Answer"] = df[index[0]].sort_values().unique()[::-1]

        # items = df[index[0]].sort_values().unique()[::-1]
        ordering = np.array(
            [
                translate(x)
                for x in np.array(list(translation.keys()))[
                    np.isin((list(translation.keys())), index)
                ]
            ]
        )
        # ordering = ordering

        # print(reordered)

        if not df[index].select_dtypes(include=[np.number]).columns.empty:
            subDF = df.loc[:, index].melt()
            subDF["variable"] = subDF["variable"].apply(lambda x: translate(x))
            trace = px.violin(
                subDF,
                y="value",
                x="variable",
                box=True,  # draw box plot inside the violin
                # points="all",
                color_discrete_sequence=["black"]
                * len(subDF.variable.unique()),  # can be 'outliers', or False
                range_y=[subDF.value.min(), subDF.value.max()],
            )

            width_of_fig = 600
            height_of_fig = 600

        else:
            subDF = df.loc[:, index].melt().value_counts()
            subDF = subDF.reset_index(level=[0, 1]).sort_values(
                ["variable", "value", 0], ascending=False
            )

            colour = sb.color_palette(
                "crest", n_colors=len(subDF["value"].unique())
            ).as_hex()
            # px.colors.sequential.deep[::2]

            subDF.rename(
                columns={0: "% Answers", "variable": "Question", "value": "Answer"},
                inplace=True,
            )

            subDF["Percentage"] = subDF.groupby("Question")["% Answers"].apply(
                lambda x: (x / len(df)) * 100
            )
            subDF["Percentage"] = subDF["Percentage"].round(3)

            subDF["Question"] = subDF["Question"].apply(lambda x: translate(x))

            if df[index[0]].dtypes.name == "category":
                if df[index[0]].cat.ordered:
                    colour = sb.diverging_palette(
                        240, 10, n=len(subDF["Answer"].unique())
                    ).as_hex()

            elif (
                df[index[0]].dtypes.name == "boolean"
                or df[index[0]].dtypes.name == "bool"
            ):
                colour = sb.diverging_palette(
                    240, 10, n=len(subDF["Answer"].unique())
                ).as_hex()

            keymap = {x: i for i, x in enumerate(ordering)}
            subDF = subDF.sort_values(
                by="Question", key=lambda series: series.apply(lambda x: keymap[x])
            )

            # print(subDF["Question"].apply(lambda x: keymap[x]))

            # reordered["Question"] : {np.arange(len(reordered["Question"]))}

            if barmode == "funnel":
                trace = px.funnel(
                    subDF,
                    x="Percentage",
                    y="Question",
                    orientation=orientation,
                    # orientation=orientation,
                    color="Answer",
                    color_discrete_sequence=colour,
                    category_orders=reordered,
                )
                trace.update_traces(textposition="none")
                trace.update_layout(xaxis={"title": r"% respondents"})

            elif barmode == "pie":

                keymap2 = {x: i for i, x in enumerate(reordered["Answer"])}
                subDF = subDF.sort_values(
                    by="Answer", key=lambda series: series.apply(lambda x: keymap2[x])
                )

                trace = px.pie(
                    subDF,
                    values="Percentage",
                    names="Answer",
                    title=subDF.Question.unique()[0],
                    # orientation="h",
                    # textinfo = 'label+percent',
                    # textposition='outside',
                    color="Answer",
                    # barmode="stack",
                    color_discrete_sequence=colour,
                )
                trace.update_layout(
                    height=400, title={"x": 0.5, "xanchor": "center"},
                )
                trace["data"][0].update(
                    {
                        "textinfo": "label+percent",
                        "textposition": "outside",
                        "showlegend": False,
                        # "sort": False,
                    }
                )
                trace.show()
                continue

            else:
                if orientation == "h":
                    trace = px.bar(
                        subDF,
                        x="Percentage",
                        y="Question",
                        orientation="h",
                        color="Answer",
                        barmode="stack",
                        color_discrete_sequence=colour,
                        category_orders=reordered,
                    )
                    trace.update_layout(xaxis={"title": r"% respondents"})

                else:
                    trace = px.bar(
                        subDF,
                        x="Question",
                        y="Percentage",
                        # orientation=orientation,
                        color="Answer",
                        barmode="stack",
                        color_discrete_sequence=colour,
                        category_orders=reordered,
                    )
                    trace.update_layout(yaxis={"title": r"% respondents"})

                if len(subDF["Question"].unique()) > max_answers:
                    max_answers = len(subDF["Question"].unique())

            width_of_fig = ncols * (100 + (75 * max_answers))

            # (
            #     len(indices) * (50 * np.max([len(x) for x in indices]))
            # ) + 200
            height_of_fig = 500

        figure1_traces = []
        for tt in range(len(trace["data"])):
            figure1_traces.append(trace["data"][tt])

        for tt in figure1_traces:
            fig.append_trace(tt, row=1, col=ii + 1)

        # width_of_fig = (len(indices) * (50 * np.max([len(x) for x in indices]))) + 200

    if barmode == "pie":
        return

    fig.update_layout(
        width=width_of_fig,
        height=height_of_fig,
        title={"text": title, "x": 0.5, "xanchor": "center",},
        # xaxis_tickangle=45,  # , "x": 0.5, "xanchor": "center"},
    )

    # width = 100 * (np.unique(df.values) * ncols)

    if orientation == "h":
        fig.update_layout(
            width=(len(indices) * 400) + 100,
            height=450,
            title={"text": title},
            # xaxis_tickangle=45,  # , "x": 0.5, "xanchor": "center"},
        )
        fig.update_layout(
            title={"x": 0.5, "xanchor": "center",}
        )

    # fig.update_layout(xaxis_type="category")
    fig.update_layout({"barmode": "stack"})
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(tickangle=45)
    fig.update_layout(showlegend=showlegend)

    if printQuestions:
        display(meta_data.style)
        return fig

    fig.show()
    return fig


#%%
def plot_map(df, title="Map"):

    fig = px.choropleth(
        df,
        locations="Country",
        #  size="Count",
        color="Count",
        locationmode="country names",
        color_continuous_scale="Aggrnyl_r",
        projection="natural earth",
        hover_name="Country",
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": dict(family="Arial", size=16),},
        # title_text='Country of Work',
        geo={
            "showcountries": True,
            "showland": True,
            "landcolor": "rgb(230, 230, 230)",
            "subunitcolor": "rgb(255, 255, 255)",
            # 'showframe':False,
            "showcoastlines": False,
            "projection_type": "natural earth",
        },
        # height=600, width = 700,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
    )
    # fig.update_layout(coloraxis_showscale=False)
    fig.show()

