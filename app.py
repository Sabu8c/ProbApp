import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sympy as sp
import math
import itertools
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

try:
    from matplotlib_venn import venn2, venn3
except ImportError:
    pass

st.set_page_config(page_title="MICRO-110 Advanced Probability Solver", layout="wide")

# Injecting premium CSS styling for custom elements, cards, and borders
st.markdown("""
<style>
    /* Global styles */
    .reportview-container {
        background: #0f172a;
    }
    /* Card design */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid #334155;
        margin-bottom: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }
    .metric-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-desc {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to render custom cards
def metric_card(title, value, description=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-header">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)

# Keyword bank for all tools
KEYWORD_BANK = {
    "Frequency & Histograms": {
        "desc": "Calculate frequency tables, relative/cumulative frequencies, and plot histograms.",
        "keywords": ["frequency", "relative frequency", "cumulative frequency", "density", "histogram", "raw data", "fraction", "observations", "less than", "greater than", "bin", "midpoint"]
    },
    "Summary Stats & Boxplots": {
        "desc": "Calculate mean, median, variance, std dev, trimmed mean, IQR, quartiles, and plot boxplots.",
        "keywords": ["mean", "median", "variance", "std dev", "standard deviation", "trimmed mean", "range", "boxplot", "quartiles", "iqr", "percentile", "outlier", "fences", "skewness", "sample variance", "population variance"]
    },
    "Set Operations (A, B, C)": {
        "desc": "Perform union, intersection, complement, differences on sets A, B, C and draw Venn diagrams.",
        "keywords": ["set", "universal", "sample space", "union", "intersection", "complement", "venn", "difference", "de morgan", "subset"]
    },
    "Sample Space Generator": {
        "desc": "Generate outcomes via Cartesian product, combinations, or permutations.",
        "keywords": ["sample space", "outcomes", "cartesian product", "combinations", "permutations", "coin flip", "dice roll"]
    },
    "Probability Rules (2- & 3-event)": {
        "desc": "Solve 2-event and 3-event probability equations and conditional rules.",
        "keywords": ["union", "intersection", "complement", "mutually exclusive", "disjoint", "independent", "conditional", "exactly one", "at least one", "neither", "both", "at most", "defect", "addition rule", "multiplication rule"]
    },
    "Total Probability & Bayes' Theorem": {
        "desc": "Calculate total probability P(E) and posterior probabilities P(Hi | E).",
        "keywords": ["bayes", "prior", "posterior", "conditional", "total probability", "hypotheses", "partitions", "given component failed", "defect", "cause", "probability tree"]
    },
    "Combinatorics & System Reliability": {
        "desc": "Calculate nCr, nPr, multiplication rules, and system reliability logic circuits.",
        "keywords": ["permutations", "combinations", "ncr", "npr", "order", "factorial", "reliability", "series", "parallel", "redundancy", "circuit", "runs", "single-configuration"]
    },
    "Custom Discrete RV": {
        "desc": "Solve custom discrete random variable PMF, E(X), Var(X), and plot CDF/PDF.",
        "keywords": ["discrete", "random variable", "rv", "pmf", "expected value", "mean", "variance", "std dev", "probability distribution"]
    },
    "Binomial Distribution": {
        "desc": "Calculate Binomial PMF, CDF, range queries, E(X), and Var(X).",
        "keywords": ["binomial", "trials", "success", "probability of success", "exact successes", "range successes", "pmf", "cdf", "discrete"]
    },
    "Poisson Distribution": {
        "desc": "Calculate Poisson probabilities, rate volume scaling, E(X), and Var(X).",
        "keywords": ["poisson", "rate", "lambda", "poisson process", "volume", "time multiplier", "expected value", "variance", "discrete"]
    },
    "Normal Distribution": {
        "desc": "Calculate Normal probabilities, Z-scores, inverse CDF (ppf), and bolt rejections.",
        "keywords": ["normal", "gaussian", "z-score", "mean", "std dev", "variance", "probability", "within k sds", "bolt rejection", "inverse cdf", "ppf", "rejections", "continuous"]
    },
    "Named Continuous (Uniform/Exp/Geometric)": {
        "desc": "Calculate Uniform, Exponential, and Geometric distributions.",
        "keywords": ["uniform", "exponential", "geometric", "continuous", "trials until success", "rate", "mean", "variance", "pdf", "cdf", "expected value"]
    },
    "Custom Continuous PDF": {
        "desc": "Parse f(x), validate area=1, calculate probabilities, E(X), Var(X), and inverse CDF.",
        "keywords": ["continuous", "pdf", "cdf", "expected value", "variance", "std dev", "integrate", "sympy", "inverse cdf", "quantile finder", "brentq", "density", "area"]
    },
    "Joint Discrete PMF": {
        "desc": "Analyze 2D joint discrete distributions, marginals, E(XY), covariance, correlation, and independence.",
        "keywords": ["joint pmf", "marginal", "covariance", "correlation", "independent", "expected value of sum", "max", "min", "overflow", "linear combination", "cdf table"]
    },
    "Joint Continuous PDF": {
        "desc": "Analyze 2D joint continuous distributions, double integrals, and independent exponentials.",
        "keywords": ["joint pdf", "double integration", "dblquad", "marginal", "expected value", "independent exponentials", "continuous", "area"]
    },
    "Covariance & Correlation": {
        "desc": "Calculate covariance and correlation coefficient from raw data or summary stats.",
        "keywords": ["covariance", "correlation", "expected value", "variance", "paired data", "scatter plot", "regression", "line", "slope"]
    },
    "Sampling Distributions & CLT": {
        "desc": "Examine sampling distribution of X̄ and calculate CLT probabilities.",
        "keywords": ["central limit theorem", "clt", "sampling distribution", "sample mean", "standard error", "se", "deviation from mean"]
    },
    "Method of Moments & MLE": {
        "desc": "Estimate parameters using MoM or MLE for Uniform, Exponential, Poisson, and Normal.",
        "keywords": ["method of moments", "mom", "maximum likelihood estimation", "mle", "unbiased estimator", "uniform", "poisson", "exponential", "normal", "log-likelihood", "parameter"]
    },
    "Confidence Intervals": {
        "desc": "Calculate Z-intervals, T-intervals, proportion intervals, and required sample size.",
        "keywords": ["confidence interval", "margin of error", "z-interval", "t-interval", "proportion", "sample size", "estimate", "precision"]
    },
    "Z-Tests (1- & 2-sample)": {
        "desc": "Perform 1-sample and 2-sample Z-tests under known population std dev.",
        "keywords": ["z-test", "hypothesis test", "mean", "known sigma", "p-value", "one-sample", "two-sample", "difference", "significance level"]
    },
    "T-Tests (1-sample, Welch, Paired)": {
        "desc": "Perform 1-sample, 2-sample Welch, and paired T-tests.",
        "keywords": ["t-test", "hypothesis test", "unknown sigma", "welch", "paired", "difference", "p-value", "before", "after", "paired differences"]
    }
}

# Sidebar Navigation
st.sidebar.title("🎲 MICRO-110 Probability Solver")
st.sidebar.markdown("---")

# Keyword Search Tool
search_query = st.sidebar.text_input("🔍 Search Keyword / Formula (e.g. 'bayes', 'trimmed'):").strip().lower()

tool = None

if search_query:
    matched_tools = []
    for tool_name, info in KEYWORD_BANK.items():
        if (search_query in tool_name.lower()) or any(search_query in kw for kw in info["keywords"]):
            matched_tools.append(tool_name)
            
    if matched_tools:
        st.sidebar.success(f"Found {len(matched_tools)} matching tools:")
        tool = st.sidebar.selectbox("Select Matched Tool", matched_tools)
        st.sidebar.info(KEYWORD_BANK[tool]["desc"])
    else:
        st.sidebar.warning("No tools match that search query. Using manual navigation instead.")

# Manual Navigation as Fallback/Default
if tool is None:
    category = st.sidebar.selectbox("Subject Area", [
        "Descriptive Stats",
        "Probability & Sets",
        "Counting",
        "Distributions",
        "Joint Distributions",
        "Inference"
    ])
    
    st.sidebar.markdown("---")
    
    if category == "Descriptive Stats":
        tool = st.sidebar.radio("Select Tool", [
            "Frequency & Histograms",
            "Summary Stats & Boxplots"
        ])
    elif category == "Probability & Sets":
        tool = st.sidebar.radio("Select Tool", [
            "Set Operations (A, B, C)",
            "Sample Space Generator",
            "Probability Rules (2- & 3-event)",
            "Total Probability & Bayes' Theorem"
        ])
    elif category == "Counting":
        tool = st.sidebar.radio("Select Tool", [
            "Combinatorics & System Reliability"
        ])
    elif category == "Distributions":
        tool = st.sidebar.radio("Select Tool", [
            "Custom Discrete RV",
            "Binomial Distribution",
            "Poisson Distribution",
            "Normal Distribution",
            "Named Continuous (Uniform/Exp/Geometric)",
            "Custom Continuous PDF"
        ])
    elif category == "Joint Distributions":
        tool = st.sidebar.radio("Select Tool", [
            "Joint Discrete PMF",
            "Joint Continuous PDF",
            "Covariance & Correlation"
        ])
    elif category == "Inference":
        tool = st.sidebar.radio("Select Tool", [
            "Sampling Distributions & CLT",
            "Method of Moments & MLE",
            "Confidence Intervals",
            "Z-Tests (1- & 2-sample)",
            "T-Tests (1-sample, Welch, Paired)"
        ])

# ==========================================
# 1. Descriptive Stats - Frequency & Histograms
# ==========================================
if tool == "Frequency & Histograms":
    st.title("Frequency & Histograms")
    st.write("Input raw data or a frequency table to calculate frequencies, relative/cumulative frequencies, and generate histograms.")

    input_type = st.radio("Input Format", ["Raw Data (comma-separated)", "Frequency Table (Values and Frequencies)"])
    
    if input_type == "Raw Data (comma-separated)":
        raw_input = st.text_area("Enter numbers separated by commas:", "10, 15, 12, 14, 18, 20, 15, 14, 13, 11")
        try:
            data = [float(x.strip()) for x in raw_input.split(",")]
            bin_method = st.radio("Binning Strategy", ["Auto", "Manual Bins"])
            if bin_method == "Manual Bins":
                st.write("Format: Start, End, Width")
                col1, col2, col3 = st.columns(3)
                start = col1.number_input("Start", value=min(data))
                end = col2.number_input("End", value=max(data) + 1.0)
                width = col3.number_input("Bin Width", value=1.0)
                bins = np.arange(start, end + width, width)
            else:
                bins = 'auto'
                
            cumulative = st.checkbox("Show Cumulative Histograms", value=False)
                
            cols = st.columns(2)
            
            with cols[0]:
                fig_freq, ax_freq = plt.subplots()
                n, b, patches = ax_freq.hist(data, bins=bins, color='skyblue', edgecolor='black', cumulative=cumulative)
                ax_freq.set_title("Cumulative Frequency" if cumulative else "Frequency Histogram")
                ax_freq.set_xlabel("Values")
                ax_freq.set_ylabel("Frequency")
                st.pyplot(fig_freq)
            
            with cols[1]:
                fig_dens, ax_dens = plt.subplots()
                ax_dens.hist(data, bins=bins, density=True, color='lightcoral', edgecolor='black', cumulative=cumulative)
                ax_dens.set_title("Cumulative Density" if cumulative else "Density Histogram")
                ax_dens.set_xlabel("Values")
                ax_dens.set_ylabel("Density")
                st.pyplot(fig_dens)
                
            st.subheader("Compute Fractions")
            threshold = st.number_input("Fraction of observations < X (Input X):", value=np.mean(data))
            fraction_less = sum([1 for d in data if d < threshold]) / len(data)
            fraction_ge = sum([1 for d in data if d >= threshold]) / len(data)
            
            c1, c2 = st.columns(2)
            with c1:
                metric_card(f"Fraction < {threshold}", f"{fraction_less:.4f}", f"Count: {sum([1 for d in data if d < threshold])} out of {len(data)}")
            with c2:
                metric_card(f"Fraction ≥ {threshold}", f"{fraction_ge:.4f}", f"Count: {sum([1 for d in data if d >= threshold])} out of {len(data)}")

        except Exception as e:
            st.error(f"Error parsing data: {e}")
            
    elif input_type == "Frequency Table (Values and Frequencies)":
        st.write("Enter the values (or bin midpoints) and their corresponding frequencies:")
        val_input = st.text_area("Values/Midpoints (comma separated):", "3.25, 3.75, 4.25, 4.75, 5.25")
        freq_input = st.text_area("Frequencies (comma separated):", "5, 15, 27, 34, 22")
        
        try:
            vals = np.array([float(x.strip()) for x in val_input.split(",")])
            freqs = np.array([float(x.strip()) for x in freq_input.split(",")])
            
            if len(vals) == len(freqs):
                total = np.sum(freqs)
                rel_freq = freqs / total
                cum_freq = np.cumsum(rel_freq)
                
                df = pd.DataFrame({
                    "Value/Midpoint": vals,
                    "Frequency": freqs,
                    "Relative Freq": rel_freq,
                    "Cumulative Freq": cum_freq
                })
                st.dataframe(df)
                
                cumulative = st.checkbox("Show Cumulative Histogram", value=False)
                graph_freqs = np.cumsum(freqs) if cumulative else freqs
                
                fig, ax = plt.subplots()
                ax.bar(vals, graph_freqs, width=(vals[1]-vals[0]) if len(vals)>1 else 1.0, color='skyblue', edgecolor='black')
                ax.set_title("Cumulative Frequency Histogram" if cumulative else "Frequency Histogram")
                st.pyplot(fig)
            else:
                st.error("Values and Frequencies must have the same length.")
        except Exception as e:
            st.error(f"Error parsing data: {e}")

# ==========================================
# 2. Descriptive Stats - Summary Stats & Boxplots
# ==========================================
elif tool == "Summary Stats & Boxplots":
    st.title("Summary Statistics & Boxplots")
    st.write("Enter multiple datasets to compare them using descriptive statistics and boxplots.")
    num_samples = st.number_input("Number of datasets", min_value=1, max_value=5, value=2)
    
    datasets = []
    labels = []
    
    for i in range(num_samples):
        st.subheader(f"Dataset {i+1}")
        col1, col2 = st.columns([1, 3])
        with col1:
            label = st.text_input(f"Label {i+1}", value=f"Sample {i+1}")
        with col2:
            raw_data = st.text_area(f"Data {i+1} (comma separated)", value="10.0, 15.0, 12.0, 14.0, 18.0, 20.0" if i==0 else "12.0, 13.0, 14.0, 15.0, 16.0, 17.0")
        try:
            data = [float(x.strip()) for x in raw_data.split(",")]
            datasets.append(data)
            labels.append(label)
        except:
            st.error(f"Invalid input in Dataset {i+1}")
            
    if len(datasets) == num_samples:
        st.write("---")
        st.subheader("Statistical Configuration")
        colA, colB, colC = st.columns(3)
        stat_mode = colA.radio("Variance Denominator", ["Sample (n-1)", "Population (n)"])
        ddof_val = 1 if "Sample" in stat_mode else 0
        
        qt_method = colB.selectbox("Quartile Interpolation (Q1/Q3)", ["linear", "lower", "higher", "midpoint", "nearest"])
        
        trim_type = colC.radio("Trimming Type", ["Percentage (%)", "Manual Count (n items)", "Auto (1.5 IQR)"], index=0)
        
        if trim_type == "Percentage (%)":
            trim_pct = st.number_input("Trimming Percentage (%) (Applies to both tails)", min_value=0.0, max_value=50.0, value=10.0)
            trim_info = f"{trim_pct}% Trimmed Mean"
        elif trim_type == "Manual Count (n items)":
            trim_count = st.number_input("Count to Trim from Top/Bottom", min_value=1, value=1)
            trim_info = f"Manual Trim ({trim_count} items/tail)"
        else:
            trim_info = "Auto Trim (Removed Outliers)"
            
        stats_list = []
        for d in datasets:
            n = len(d)
            d_sorted = np.sort(d)
            mean = np.mean(d)
            median = np.median(d)
            
            # Quartile calculation
            q1, q3 = np.percentile(d, [25, 75], method=qt_method) if n > 0 else (0, 0)
            iqr = q3 - q1
            low_f, up_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = [x for x in d if x < low_f or x > up_f]
            
            # Trimming Logic
            if trim_type == "Percentage (%)":
                trim_m = stats.trim_mean(d, proportiontocut=trim_pct/100.0)
            elif trim_type == "Manual Count (n items)":
                if n > (2 * trim_count):
                    trimmed_data = d_sorted[trim_count:-trim_count]
                    trim_m = np.mean(trimmed_data)
                else: trim_m = 0.0
            else: # Auto
                trimmed_data = [x for x in d if low_f <= x <= up_f]
                trim_m = np.mean(trimmed_data) if trimmed_data else 0.0
            
            skew = "Symmetric" if np.isclose(mean, median, atol=1e-5) else ("Positive" if mean > median else "Negative")
            
            sample_var = np.var(d, ddof=ddof_val) if n > ddof_val else 0
            sample_std = np.std(d, ddof=ddof_val) if n > ddof_val else 0
            rng = np.max(d) - np.min(d)
            
            stats_list.append({
                "N": n,
                "Mean": mean,
                "Median": median,
                "Skewness": skew,
                trim_info: trim_m,
                "IQR": iqr,
                "Q1": q1,
                "Q3": q3,
                "Fences": f"[{low_f:.2f}, {up_f:.2f}]",
                "Outliers (#)": len(outliers),
                "Variance": sample_var,
                "Std Dev": sample_std,
                "Range": rng
            })
            
        st.dataframe(pd.DataFrame(stats_list, index=labels))
        
        st.subheader("Comparative Boxplot")
        fig, ax = plt.subplots()
        ax.boxplot(datasets, labels=labels)
        ax.set_ylabel("Values")
        ax.set_title("Boxplot Comparison")
        st.pyplot(fig)

# ==========================================
# 3. Set Operations (A, B, C)
# ==========================================
elif tool == "Set Operations (A, B, C)":
    st.title("Set Operations & Venn Diagrams")
    st.write("Compute unions, intersections, complements, and differences for up to three sets.")
    
    def parse_set(s):
        return set([x.strip() for x in s.split(",") if x.strip()])
        
    s_input = st.text_area("Universal Sample Space S (comma separated):", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    a_input = st.text_input("Event A (comma separated):", "2, 4, 6, 8, 10")
    b_input = st.text_input("Event B (comma separated):", "6, 7, 8, 9, 10")
    c_input = st.text_input("Event C (comma separated, optional):", "1, 2, 3, 7, 8")
    
    S = parse_set(s_input)
    A = parse_set(a_input)
    B = parse_set(b_input)
    C = parse_set(c_input)
    
    st.write(f"**S:** {S} (Size: {len(S)})")
    st.write(f"**A:** {A} (Size: {len(A)})")
    st.write(f"**B:** {B} (Size: {len(B)})")
    if C:
        st.write(f"**C:** {C} (Size: {len(C)})")
        
    st.subheader("Core Set Algebra Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**A ∪ B (Union of A & B):** `{A.union(B)}`")
        st.markdown(f"**A ∩ B (Intersection of A & B):** `{A.intersection(B)}`")
        st.markdown(f"**A' (Complement of A):** `{S.difference(A)}`")
        st.markdown(f"**B' (Complement of B):** `{S.difference(B)}`")
        st.markdown(f"**(A ∪ B)' (Complement of Union):** `{S.difference(A.union(B))}`")
        
    with col2:
        if C:
            st.markdown(f"**A ∪ B ∪ C (Union of A, B, C):** `{A.union(B).union(C)}`")
            st.markdown(f"**A ∩ B ∩ C (Intersection of A, B, C):** `{A.intersection(B).intersection(C)}`")
            st.markdown(f"**C' (Complement of C):** `{S.difference(C)}`")
            st.markdown(f"**(A ∪ B ∪ C)' (Complement of Union A, B, C):** `{S.difference(A.union(B).union(C))}`")
            st.markdown(f"**(A ∩ B ∩ C') (A and B but not C):** `{A.intersection(B).difference(C)}`")
            
    # Venn diagram rendering
    if 'venn2' in globals():
        st.subheader("Venn Diagram Visualization")
        fig, ax = plt.subplots(figsize=(6, 4))
        if C:
            # Calculate intersection sizes
            s_abc = len(A.intersection(B).intersection(C))
            s_ab_nc = len(A.intersection(B).difference(C))
            s_ac_nb = len(A.intersection(C).difference(B))
            s_bc_na = len(B.intersection(C).difference(A))
            s_a_only = len(A.difference(B).difference(C))
            s_b_only = len(B.difference(A).difference(C))
            s_c_only = len(C.difference(A).difference(B))
            
            try:
                venn3(subsets=(s_a_only, s_b_only, s_ab_nc, s_c_only, s_ac_nb, s_bc_na, s_abc),
                      set_labels=('A', 'B', 'C'), ax=ax)
                st.pyplot(fig)
            except:
                pass
        else:
            s_a_only = len(A.difference(B))
            s_b_only = len(B.difference(A))
            s_ab = len(A.intersection(B))
            try:
                venn2(subsets=(s_a_only, s_b_only, s_ab), set_labels=('A', 'B'), ax=ax)
                st.pyplot(fig)
            except:
                pass

# ==========================================
# 4. Sample Space Generator
# ==========================================
elif tool == "Sample Space Generator":
    st.title("Sample Space Generator")
    st.write("Generate and list all exact possible outcomes for your sample space.")
    
    gen_type = st.radio("Generation Type", [
        "Cartesian Product (e.g. flipping multiple coins, rolling dice)", 
        "Combinations (Choosing items, order doesn't matter)", 
        "Permutations (Arranging items, order matters)"
    ])
    
    if "Cartesian" in gen_type:
        st.write("Enter the possible outcomes for each independent event/category.")
        num_events = st.number_input("Number of Events / Categories", min_value=1, max_value=7, value=2)
        events = []
        for i in range(num_events):
            event_input = st.text_input(f"Event {i+1} Outcomes (comma separated)", value="H, T" if i==0 else "1, 2, 3, 4, 5, 6")
            events.append([x.strip() for x in event_input.split(",") if x.strip()])
            
        if st.button("Generate Sample Space"):
            outcomes = list(itertools.product(*events))
            st.success(f"Total Outcomes: {len(outcomes)}")
            st.write(["(" + ", ".join(str(item) for item in o) + ")" for o in outcomes])
            
    else:
        items_input = st.text_input("Items to choose from (comma separated)", "A, B, C, D")
        items = [x.strip() for x in items_input.split(",") if x.strip()]
        
        max_k = len(items) if len(items) > 0 else 1
        default_k = min(2, max_k)
        
        k = st.number_input("How many to choose at a time (k)", min_value=1, max_value=max_k, value=default_k)
        
        if st.button("Generate Outcomes"):
            if "Combinations" in gen_type:
                outcomes = list(itertools.combinations(items, k))
            else:
                outcomes = list(itertools.permutations(items, k))
            
            st.success(f"Total Outcomes: {len(outcomes)}")
            st.write(["(" + ", ".join(str(item) for item in o) + ")" for o in outcomes])

# ==========================================
# 5. Probability Rules (2- & 3-event)
# ==========================================
elif tool == "Probability Rules (2- & 3-event)":
    st.title("Probability Rules Solver")
    
    scheme = st.radio("Select Scheme", ["Two Events (A, B)", "Three Events (A1, A2, A3)"])
    
    if scheme == "Two Events (A, B)":
        input_mode = st.selectbox("Input Mode", [
            "Given P(A), P(B), and P(A ∪ B)",
            "Given P(A), P(B), and P(A ∩ B)",
            "Given P(A), P(B), and P(B|A)"
        ])
        
        col1, col2 = st.columns(2)
        p_a = col1.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.55)
        p_b = col2.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.45)
        
        if input_mode == "Given P(A), P(B), and P(A ∪ B)":
            p_a_or_b = st.number_input("P(A ∪ B)", min_value=0.0, max_value=1.0, value=0.70)
            p_a_and_b = p_a + p_b - p_a_or_b
        elif input_mode == "Given P(A), P(B), and P(A ∩ B)":
            p_a_and_b = st.number_input("P(A ∩ B)", min_value=0.0, max_value=1.0, value=0.30)
            p_a_or_b = p_a + p_b - p_a_and_b
        else:
            p_b_given_a = st.number_input("P(B|A)", min_value=0.0, max_value=1.0, value=0.50)
            p_a_and_b = p_b_given_a * p_a
            p_a_or_b = p_a + p_b - p_a_and_b
            
        p_not_a_and_not_b = 1 - p_a_or_b
        
        st.subheader("Results")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.write(f"**P(A ∩ B) (Both A and B):** {p_a_and_b:.4f}")
            st.write(f"**P(A ∪ B) (At least one of A or B):** {p_a_or_b:.4f}")
            st.write(f"**P(A' ∩ B') (Neither A nor B):** {p_not_a_and_not_b:.4f}")
            st.write(f"**P(Exactly one of A or B):** {p_a_or_b - p_a_and_b:.4f}")
            
        with col_res2:
            if p_a > 0:
                st.write(f"**P(B|A) (B given A):** {p_a_and_b / p_a:.4f}")
                st.write(f"**P(B'|A) (not B given A):** {1 - (p_a_and_b / p_a):.4f}")
            if p_b > 0:
                st.write(f"**P(A|B) (A given B):** {p_a_and_b / p_b:.4f}")
                st.write(f"**P(A'|B) (not A given B):** {1 - (p_a_and_b / p_b):.4f}")
            st.write(f"**P(A ∩ B') (A but not B):** {p_a - p_a_and_b:.4f}")
            st.write(f"**P(B ∩ A') (B but not A):** {p_b - p_a_and_b:.4f}")
            
        try:
            fig, ax = plt.subplots(figsize=(4, 3))
            v = venn2(subsets=(p_a - p_a_and_b, p_b - p_a_and_b, p_a_and_b), set_labels=('A', 'B'), ax=ax)
            if v and v.get_label_by_id('10'): v.get_label_by_id('10').set_text(f"{max(0.0, p_a - p_a_and_b):.2f}")
            if v and v.get_label_by_id('01'): v.get_label_by_id('01').set_text(f"{max(0.0, p_b - p_a_and_b):.2f}")
            if v and v.get_label_by_id('11'): v.get_label_by_id('11').set_text(f"{max(0.0, p_a_and_b):.2f}")
            st.pyplot(fig)
        except Exception:
            pass
            
    elif scheme == "Three Events (A1, A2, A3)":
        col1, col2 = st.columns(2)
        p_a1 = col1.number_input("P(A1)", value=0.12)
        p_a2 = col1.number_input("P(A2)", value=0.07)
        p_a3 = col1.number_input("P(A3)", value=0.05)
        
        p_a1_or_a2 = col2.number_input("P(A1 ∪ A2)", value=0.13)
        p_a1_or_a3 = col2.number_input("P(A1 ∪ A3)", value=0.14)
        p_a2_or_a3 = col2.number_input("P(A2 ∪ A3)", value=0.10)
        
        p_all_3 = st.number_input("P(A1 ∩ A2 ∩ A3)", value=0.01)
        
        p_a1_and_a2 = p_a1 + p_a2 - p_a1_or_a2
        p_a1_and_a3 = p_a1 + p_a3 - p_a1_or_a3
        p_a2_and_a3 = p_a2 + p_a3 - p_a2_or_a3
        
        p_at_least_one = p_a1 + p_a2 + p_a3 - p_a1_and_a2 - p_a1_and_a3 - p_a2_and_a3 + p_all_3
        p_at_least_two = p_a1_and_a2 + p_a1_and_a3 + p_a2_and_a3 - 2 * p_all_3
        p_exactly_one = p_at_least_one - p_at_least_two
        p_exactly_two = p_at_least_two - p_all_3
        p_neither = 1 - p_at_least_one
        p_at_most_two = 1 - p_all_3
        
        st.subheader("Exact Probabilities & Set Partitions")
        
        df_probs = pd.DataFrame({
            "Metric / Event Combination": [
                "P(At least one defect / Union of all)",
                "P(At least two defects)",
                "P(Exactly one defect)",
                "P(Exactly two defects)",
                "P(All three defects)",
                "P(No defects / Neither)",
                "P(At most two defects)"
            ],
            "Probability": [
                p_at_least_one, p_at_least_two, p_exactly_one, p_exactly_two, p_all_3, p_neither, p_at_most_two
            ]
        })
        st.dataframe(df_probs.style.format({"Probability": "{:.6f}"}))
        
        st.subheader("Conditional Queries")
        st.write(f"**P(Exactly one | At least one):** {p_exactly_one / p_at_least_one if p_at_least_one > 0 else 0.0:.6f}")
        st.write(f"**P(All Three | A1):** {p_all_3 / p_a1 if p_a1 > 0 else 0.0:.6f}")
        st.write(f"**P(Not A3 | A1 ∩ A2):** {(p_a1_and_a2 - p_all_3) / p_a1_and_a2 if p_a1_and_a2 > 0 else 0.0:.6f}")
        
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            v = venn3(subsets=(
                max(0.0, p_a1 - p_a1_and_a2 - p_a1_and_a3 + p_all_3),
                max(0.0, p_a2 - p_a1_and_a2 - p_a2_and_a3 + p_all_3),
                max(0.0, p_a1_and_a2 - p_all_3),
                max(0.0, p_a3 - p_a1_and_a3 - p_a2_and_a3 + p_all_3),
                max(0.0, p_a1_and_a3 - p_all_3),
                max(0.0, p_a2_and_a3 - p_all_3),
                max(0.0, p_all_3)
            ), set_labels=('A1', 'A2', 'A3'), ax=ax)
            if v: st.pyplot(fig)
        except Exception:
            pass

# ==========================================
# 6. Total Probability & Bayes' Theorem
# ==========================================
elif tool == "Total Probability & Bayes' Theorem":
    st.title("Total Probability & Bayes' Theorem Solver")
    st.write("Solve problems where an event B occurs under different partitions/hypotheses (H1, H2, ..., Hn).")
    
    num_hyp = st.number_input("Number of partitions / hypotheses (n)", min_value=2, max_value=10, value=3)
    
    hyp_names = []
    priors = []
    conditionals = []
    
    cols = st.columns(num_hyp)
    for i in range(num_hyp):
        with cols[i]:
            name = st.text_input(f"Hypothesis Name {i+1}", value=f"H{i+1}")
            prior = st.number_input(f"P({name})", min_value=0.0, max_value=1.0, value=1.0/num_hyp, format="%.4f")
            cond = st.number_input(f"P(Event | {name})", min_value=0.0, max_value=1.0, value=0.5, format="%.4f")
            hyp_names.append(name)
            priors.append(prior)
            conditionals.append(cond)
            
    priors = np.array(priors)
    conditionals = np.array(conditionals)
    
    # Check sum of priors
    sum_priors = np.sum(priors)
    if not np.isclose(sum_priors, 1.0, atol=1e-4):
        st.warning(f"Warning: Priors sum to {sum_priors:.4f} instead of 1.0. Normalizing priors for calculations...")
        priors = priors / sum_priors
        
    # Total Probability
    p_event = np.sum(priors * conditionals)
    
    # Posteriors
    posteriors = (priors * conditionals) / p_event if p_event > 0 else np.zeros_like(priors)
    
    st.subheader("Calculation Summary")
    
    df_bayes = pd.DataFrame({
        "Hypothesis": hyp_names,
        "Prior P(H_i)": priors,
        "Conditional P(E | H_i)": conditionals,
        "Joint P(E ∩ H_i)": priors * conditionals,
        "Posterior P(H_i | E)": posteriors
    })
    
    st.dataframe(df_bayes.style.format({
        "Prior P(H_i)": "{:.6f}",
        "Conditional P(E | H_i)": "{:.6f}",
        "Joint P(E ∩ H_i)": "{:.6f}",
        "Posterior P(H_i | E)": "{:.6f}"
    }))
    
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Total Probability P(Event)", f"{p_event:.6f}", "Sum of all Joint Probabilities")
    with c2:
        max_idx = np.argmax(posteriors)
        metric_card("Most Likely Cause", f"{hyp_names[max_idx]}", f"Posterior Probability: {posteriors[max_idx]:.6f}")

# ==========================================
# 7. Combinatorics & System Reliability
# ==========================================
elif tool == "Combinatorics & System Reliability":
    st.title("Combinatorics & System Reliability")
    
    section = st.radio("Section Selection", ["Combinatorics (Counting)", "System Reliability"])
    
    if section == "Combinatorics (Counting)":
        st.subheader("Equipment Single-Configuration Runs")
        num_categories = st.number_input("Number of categories (e.g. Temp, Pressure, Catalyst)", min_value=1, value=3)
        counts = []
        for i in range(num_categories):
            counts.append(st.number_input(f"Number of options in category {i+1}", min_value=1, value=[3,4,5][i] if i<3 else 2))
            
        runs = np.prod(counts)
        st.write(f"**Total single-configuration runs possible:** {runs}")
        
        st.write("---")
        st.subheader("Permutations & Combinations (nPk / nCk)")
        n_val = st.number_input("n (total items)", min_value=1, value=10)
        k_val = st.number_input("k (items to select)", min_value=0, max_value=n_val, value=3)
        
        nCr = math.comb(n_val, k_val)
        nPr = math.perm(n_val, k_val)
        st.write(f"**Combinations ({n_val}C{k_val}):** {nCr}")
        st.write(f"**Permutations ({n_val}P{k_val}):** {nPr}")
        
    else:
        st.subheader("System Reliability Calculator")
        st.write("Compute system working probability using boolean circuit combinations.")
        st.write("Use `&` for components in **Series** (multiplication of probabilities).")
        st.write("Use `|` for components in **Parallel** (1 - product of failure probabilities).")
        
        # User defined number of components
        num_comp = st.number_input("Number of components to define", min_value=1, max_value=20, value=5)
        
        p_dict = {}
        cols = st.columns(min(num_comp, 5))
        for i in range(num_comp):
            c_idx = i % 5
            with cols[c_idx]:
                val = st.number_input(f"P{i+1}", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
                p_dict[f"P{i+1}"] = val
                
        expression = st.text_input("System Expression (e.g. `(P1 | P2) & P3 & (P4 | P5)`)", "(P1 | P2) & P3 & (P4 | P5)")
        
        def eval_prob_expr(expr, comp_probs):
            class ProbNode:
                def __init__(self, val):
                    self.val = val
                def __and__(self, other):
                    return ProbNode(self.val * other.val)
                def __or__(self, other):
                    return ProbNode(1 - (1 - self.val) * (1 - other.val))
                    
            env = {k: ProbNode(v) for k, v in comp_probs.items()}
            try:
                result = eval(expr, {}, env)
                return result.val
            except Exception as e:
                st.error(f"Error evaluating expression: {e}")
                return None
                
        if expression:
            res = eval_prob_expr(expression, p_dict)
            if res is not None:
                st.success(f"**System Reliability Probability:** {res:.6f}")

# ==========================================
# 8. Custom Discrete RV
# ==========================================
elif tool == "Custom Discrete RV":
    st.title("Custom Discrete Random Variable Solver")
    st.write("Input probabilities for custom discrete values to compute expected value, variance, and plot distributions.")
    
    val_input = st.text_input("Values for X (comma separated):", "-1, 0, 1, 2")
    prob_input = st.text_input("Probabilities P(X) (comma separated):", "0.1, 0.4, 0.3, 0.2")
    
    try:
        x_vals = np.array([float(x.strip()) for x in val_input.split(",")])
        p_vals = np.array([float(x.strip()) for x in prob_input.split(",")])
        
        if len(x_vals) == len(p_vals):
            total_p = np.sum(p_vals)
            st.write(f"**Sum of Probabilities:** {total_p:.4f} (Normally should be 1.0)")
            
            ex = np.sum(x_vals * p_vals)
            ex2 = np.sum((x_vals**2) * p_vals)
            vx = ex2 - (ex**2)
            stdx = np.sqrt(vx) if vx >= 0 else 0
            
            col_1, col_2 = st.columns(2)
            with col_1:
                metric_card("Expected Value E(X)", f"{ex:.4f}")
                metric_card("E(X^2)", f"{ex2:.4f}")
            with col_2:
                metric_card("Variance Var(X)", f"{vx:.4f}")
                metric_card("Std Dev σ", f"{stdx:.4f}")
                
            cumulative = st.checkbox("Show Cumulative Distribution", value=False)
            graph_p = np.cumsum(p_vals) if cumulative else p_vals
            
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(x_vals, graph_p, color='orange', edgecolor='black', width=(max(x_vals)-min(x_vals))/max(10, len(x_vals)) if len(x_vals)>1 else 0.5)
            ax.set_title("Cumulative Distribution" if cumulative else "Discrete Probability Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("P(X <= x)" if cumulative else "P(X)")
            st.pyplot(fig)
        else:
            st.error("Values and Probabilities must have the same length.")
    except Exception as e:
        st.error(f"Error computing distribution: {e}")

# ==========================================
# 9. Binomial Distribution
# ==========================================
elif tool == "Binomial Distribution":
    st.title("Binomial Distribution")
    
    n_binom = st.number_input("n (Number of Trials)", min_value=1, value=10)
    p_binom = st.number_input("p (Probability of Success)", min_value=0.0, max_value=1.0, value=0.25)
    
    st.write(f"**X ~ Bin({n_binom}, {p_binom})**")
    
    query = st.selectbox("Query Type", ["Exact Successes P(X = x)", "Range Query P(a <= X <= b)"])
    
    if query == "Exact Successes P(X = x)":
        x_val = st.number_input("x (Target Successes)", min_value=0, max_value=n_binom, value=2)
        prob_exact = stats.binom.pmf(x_val, n_binom, p_binom)
        prob_le = stats.binom.cdf(x_val, n_binom, p_binom)
        prob_ge = 1 - stats.binom.cdf(x_val - 1, n_binom, p_binom) if x_val > 0 else 1.0
        
        st.write(f"**P(X = {x_val}):** {prob_exact:.6f}")
        st.write(f"**P(X <= {x_val}):** {prob_le:.6f}")
        st.write(f"**P(X >= {x_val}):** {prob_ge:.6f}")
    else:
        col1, col2 = st.columns(2)
        a_val = col1.number_input("a (lower bound)", min_value=0, max_value=n_binom, value=1)
        b_val = col2.number_input("b (upper bound)", min_value=0, max_value=n_binom, value=4)
        
        if a_val <= b_val:
            prob_range = stats.binom.cdf(b_val, n_binom, p_binom) - stats.binom.cdf(a_val - 1, n_binom, p_binom)
            st.success(f"**P({a_val} <= X <= {b_val}):** {prob_range:.6f}")
        else:
            st.error("Lower bound 'a' must be less than or equal to upper bound 'b'.")
            
    mean_v = stats.binom.mean(n_binom, p_binom)
    var_v = stats.binom.var(n_binom, p_binom)
    
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Expected Value E(X)", f"{mean_v:.4f}")
    with c2:
        metric_card("Variance Var(X)", f"{var_v:.4f}")
        
    cumulative = st.checkbox("Show Cumulative Distribution", value=False)
    xs = np.arange(0, n_binom + 1)
    pmfs = stats.binom.pmf(xs, n_binom, p_binom)
    cmfs = stats.binom.cdf(xs, n_binom, p_binom)
    fig, ax = plt.subplots()
    ax.bar(xs, cmfs if cumulative else pmfs, color='lightblue', edgecolor='black')
    ax.set_title("Binomial CDF" if cumulative else "Binomial PMF")
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability" if cumulative else "Probability")
    st.pyplot(fig)

# ==========================================
# 10. Poisson Distribution
# ==========================================
elif tool == "Poisson Distribution":
    st.title("Poisson Distribution")
    
    lam_base = st.number_input("Base rate λ (per unit)", value=5.0, min_value=0.0001)
    volume = st.number_input("Volume / time multiplier", value=1.0, min_value=0.0001)
    lam = lam_base * volume
    st.write(f"**Effective λ = {lam_base} × {volume} = {lam:.4f}**")
    
    query_p = st.selectbox("Query", [
        "P(X = k)", "P(X ≤ k)", "P(X ≥ k)",
        "P(a ≤ X ≤ b)", "P(X > μ + 1·SD)", "Find volume for P(X ≥ 1) = target"
    ])
    
    if query_p == "P(X = k)":
        k = st.number_input("k", value=3, min_value=0)
        st.write(f"**P(X = {k}) = {stats.poisson.pmf(k, lam):.6f}**")
    elif query_p == "P(X ≤ k)":
        k = st.number_input("k", value=3, min_value=0)
        st.write(f"**P(X ≤ {k}) = {stats.poisson.cdf(k, lam):.6f}**")
    elif query_p == "P(X ≥ k)":
        k = st.number_input("k", value=3, min_value=0)
        p = 1 - stats.poisson.cdf(k - 1, lam) if k > 0 else 1.0
        st.write(f"**P(X ≥ {k}) = {p:.6f}**")
    elif query_p == "P(a ≤ X ≤ b)":
        col1, col2 = st.columns(2)
        a_val = col1.number_input("a", min_value=0, value=2)
        b_val = col2.number_input("b", min_value=0, value=5)
        p = stats.poisson.cdf(b_val, lam) - stats.poisson.cdf(a_val - 1, lam) if a_val > 0 else stats.poisson.cdf(b_val, lam)
        st.write(f"**P({a_val} ≤ X ≤ {b_val}) = {p:.6f}**")
    elif query_p == "P(X > μ + 1·SD)":
        mean_v = lam
        sd_v = np.sqrt(lam)
        threshold = mean_v + sd_v
        k_thresh = int(np.floor(threshold))
        p = 1 - stats.poisson.cdf(k_thresh, lam)
        st.write(f"**μ = {mean_v:.4f}, σ = {sd_v:.4f}, μ+σ = {threshold:.4f}**")
        st.write(f"**P(X > {k_thresh}) = {p:.6f}**")
    elif query_p == "Find volume for P(X ≥ 1) = target":
        target = st.number_input("Target probability", value=0.999, min_value=0.0001, max_value=0.9999)
        v_needed = -np.log(1 - target) / lam_base
        st.write(f"**Required volume = {v_needed:.6f}**")
        
    st.write(f"**E(X) = {lam:.4f},  Var(X) = {lam:.4f},  SD = {np.sqrt(lam):.4f}**")
    
    xs_p = np.arange(0, int(lam + 4*np.sqrt(lam)) + 1)
    fig, ax = plt.subplots()
    ax.bar(xs_p, stats.poisson.pmf(xs_p, lam), color='lightblue', edgecolor='black')
    ax.set_title(f"Poisson(λ={lam:.2f}) PMF")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X=k)")
    st.pyplot(fig)

# ==========================================
# 11. Normal Distribution
# ==========================================
elif tool == "Normal Distribution":
    st.title("Normal Distribution Calculator")
    st.write("Compute probabilities for $X \\sim N(\\mu, \\sigma^2)$.")
    
    col1, col2 = st.columns(2)
    mu = col1.number_input("Mean (μ)", value=0.0)
    sigma = col2.number_input("Std Dev (σ)", value=1.0, min_value=0.0001)
    
    query = st.selectbox("Query Type", [
        "P(X < a)", "P(X > a)", "P(a < X < b)",
        "P(X < a OR X > b)", "P(within k SDs of mean)",
        "Find x for given cumulative probability", "Bolt rejection (count short/long)"
    ])
    
    if query == "P(X < a)":
        a = st.number_input("a", value=1.96)
        p = stats.norm.cdf(a, mu, sigma)
        st.write(f"**z = {(a - mu)/sigma:.4f}**")
        st.write(f"**P(X < {a}) = {p:.6f}**")
    elif query == "P(X > a)":
        a = st.number_input("a", value=1.96)
        p = 1 - stats.norm.cdf(a, mu, sigma)
        st.write(f"**z = {(a - mu)/sigma:.4f}**")
        st.write(f"**P(X > {a}) = {p:.6f}**")
    elif query == "P(a < X < b)":
        c1, c2 = st.columns(2)
        a = c1.number_input("a (lower)", value=-1.96)
        b = c2.number_input("b (upper)", value=1.96)
        p = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)
        st.write(f"**P({a} < X < {b}) = {p:.6f}**")
    elif query == "P(X < a OR X > b)":
        c1, c2 = st.columns(2)
        a = c1.number_input("a (lower tail)", value=-1.96)
        b = c2.number_input("b (upper tail)", value=1.96)
        p = stats.norm.cdf(a, mu, sigma) + (1 - stats.norm.cdf(b, mu, sigma))
        st.write(f"**P(X < {a} or X > {b}) = {p:.6f}**")
    elif query == "P(within k SDs of mean)":
        k = st.number_input("k (number of SDs)", value=1.0, min_value=0.0)
        p = stats.norm.cdf(mu + k*sigma, mu, sigma) - stats.norm.cdf(mu - k*sigma, mu, sigma)
        st.write(f"**P(|X − μ| ≤ {k}σ) = P({mu - k*sigma:.4f} < X < {mu + k*sigma:.4f}) = {p:.6f}**")
        st.write(f"**P(|X − μ| > {k}σ) = {1 - p:.6f}**")
    elif query == "Find x for given cumulative probability":
        p_target = st.number_input("Cumulative probability P(X ≤ x)", value=0.95, min_value=0.0001, max_value=0.9999)
        x_val = stats.norm.ppf(p_target, mu, sigma)
        st.write(f"**x = {x_val:.6f}**")
    elif query == "Bolt rejection (count short/long)":
        st.write("Given N items from N(μ,σ), count expected rejects outside [lower, upper].")
        c1, c2, c3 = st.columns(3)
        lower = c1.number_input("Lower limit", value=-2.0)
        upper = c2.number_input("Upper limit", value=2.0)
        N = c3.number_input("Total items N", value=1000, min_value=1)
        p_short = stats.norm.cdf(lower, mu, sigma)
        p_long = 1 - stats.norm.cdf(upper, mu, sigma)
        st.write(f"**P(X < {lower}) = {p_short:.6f}  →  n_short ≈ {p_short * N:.1f}**")
        st.write(f"**P(X > {upper}) = {p_long:.6f}  →  n_long ≈ {p_long * N:.1f}**")
        st.write(f"**Expected accepted: {N - p_short*N - p_long*N:.1f}**")
        
    # Plot
    xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    ys = stats.norm.pdf(xs, mu, sigma)
    fig, ax = plt.subplots()
    ax.plot(xs, ys, 'b-')
    ax.fill_between(xs, 0, ys, alpha=0.1, color='blue')
    ax.set_title(f"N({mu}, {sigma}²)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    st.pyplot(fig)

# ==========================================
# 12. Named Continuous (Uniform/Exp/Geometric)
# ==========================================
elif tool == "Named Continuous (Uniform/Exp/Geometric)":
    st.title("Named Distributions Solver")
    
    dist_choice = st.selectbox("Select Distribution", ["Uniform (Continuous)", "Exponential", "Geometric"])
    
    if dist_choice == "Uniform (Continuous)":
        col1, col2 = st.columns(2)
        a_bound = col1.number_input("Lower limit a", value=0.0)
        b_bound = col2.number_input("Upper limit b", value=10.0)
        
        if a_bound < b_bound:
            st.write(f"**X ~ Uniform({a_bound}, {b_bound})**")
            ex = (a_bound + b_bound) / 2
            vx = ((b_bound - a_bound) ** 2) / 12
            
            c1, c2 = st.columns(2)
            with c1:
                metric_card("Expected Value E(X)", f"{ex:.6f}")
            with c2:
                metric_card("Variance Var(X)", f"{vx:.6f}")
                
            st.subheader("Query P(x1 < X < x2)")
            colA, colB = st.columns(2)
            x1 = colA.number_input("x1", value=a_bound)
            x2 = colB.number_input("x2", value=(a_bound + b_bound)/2)
            
            c_x1 = max(a_bound, min(b_bound, x1))
            c_x2 = max(a_bound, min(b_bound, x2))
            
            prob = (c_x2 - c_x1) / (b_bound - a_bound) if c_x2 > c_x1 else 0.0
            st.success(f"**P({x1} < X < {x2}) = {prob:.6f}**")
        else:
            st.error("Lower bound 'a' must be less than upper bound 'b'.")
            
    elif dist_choice == "Exponential":
        input_mode = st.radio("Input mode", ["Rate λ", "Mean β (1/λ)"])
        if input_mode == "Rate λ":
            lam = st.number_input("Rate parameter λ", value=0.5, min_value=0.0001)
            beta = 1.0 / lam
        else:
            beta = st.number_input("Mean parameter β", value=2.0, min_value=0.0001)
            lam = 1.0 / beta
            
        st.write(f"**X ~ Exp(λ={lam:.4f})**")
        
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Expected Value E(X) = 1/λ", f"{beta:.6f}")
        with c2:
            metric_card("Variance Var(X) = 1/λ²", f"{(beta**2):.6f}")
            
        query = st.selectbox("Query", ["P(X < x)", "P(X > x)", "P(x1 < X < x2)"])
        if query == "P(X < x)":
            x_val = st.number_input("x", value=1.0)
            p = 1 - np.exp(-lam * x_val) if x_val >= 0 else 0.0
            st.success(f"**P(X < {x_val}) = {p:.6f}**")
        elif query == "P(X > x)":
            x_val = st.number_input("x", value=1.0)
            p = np.exp(-lam * x_val) if x_val >= 0 else 1.0
            st.success(f"**P(X > {x_val}) = {p:.6f}**")
        else:
            c_low, c_high = st.columns(2)
            x1 = c_low.number_input("x1", value=1.0)
            x2 = c_high.number_input("x2", value=3.0)
            p = (np.exp(-lam * x1) - np.exp(-lam * x2)) if x2 > x1 else 0.0
            st.success(f"**P({x1} < X < {x2}) = {p:.6f}**")
            
    elif dist_choice == "Geometric":
        p_geom = st.number_input("Probability of success p", min_value=0.0001, max_value=1.0, value=0.3)
        st.write("**X = Number of trials until the first success (starts at 1)**")
        
        ex = 1.0 / p_geom
        vx = (1.0 - p_geom) / (p_geom ** 2)
        
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Expected Value E(X) = 1/p", f"{ex:.6f}")
        with c2:
            metric_card("Variance Var(X) = (1-p)/p²", f"{vx:.6f}")
            
        query = st.selectbox("Query", ["P(X = k)", "P(X <= k)", "P(X >= k)"])
        k = st.number_input("k", min_value=1, value=3)
        
        if query == "P(X = k)":
            p = ((1.0 - p_geom) ** (k - 1)) * p_geom
            st.success(f"**P(X = {k}) = {p:.6f}**")
        elif query == "P(X <= k)":
            p = 1.0 - ((1.0 - p_geom) ** k)
            st.success(f"**P(X <= {k}) = {p:.6f}**")
        else:
            p = (1.0 - p_geom) ** (k - 1)
            st.success(f"**P(X >= {k}) = {p:.6f}**")

# ==========================================
# 13. Custom Continuous PDF
# ==========================================
elif tool == "Custom Continuous PDF":
    st.title("Custom Continuous Random Variables")
    st.write("Input the PDF $f(x)$ as a python/sympy expression (e.g. `1.5*(1-x**2)`). Use `x` as the variable.")
    
    func_str = st.text_input("f(x) =", "1.5 * (1 - x**2)")
    col1, col2 = st.columns(2)
    a_bound = col1.number_input("Lower Bound (a)", value=-1.0)
    b_bound = col2.number_input("Upper Bound (b)", value=1.0)
    
    try:
        x_sym = sp.Symbol('x')
        f_sym = sp.sympify(func_str)
        f_lamb = sp.lambdify(x_sym, f_sym, 'numpy')
        
        total_prob, _ = quad(f_lamb, a_bound, b_bound)
        st.write(f"**Total Area (should be ~1.0):** {total_prob:.6f}")
        
        st.markdown("---")
        query_mode = st.radio("Calculation Type", ["Compute Probability", "Expected Value / SD Bounds", "Inverse CDF (Quantile Finder)"])
        
        if query_mode == "Compute Probability":
            colA, colB = st.columns(2)
            x1 = colA.number_input("Lower Limit", value=0.0)
            x2 = colB.number_input("Upper Limit", value=0.5)
            
            prob_range, _ = quad(f_lamb, max(a_bound, x1), min(b_bound, x2))
            st.success(f"**P({x1} < X < {x2}):** {prob_range:.6f}")
            
        elif query_mode == "Expected Value / SD Bounds":
            ex_lamb = sp.lambdify(x_sym, x_sym * f_sym, 'numpy')
            ex, _ = quad(ex_lamb, a_bound, b_bound)
            
            ex2_lamb = sp.lambdify(x_sym, (x_sym**2) * f_sym, 'numpy')
            ex2, _ = quad(ex2_lamb, a_bound, b_bound)
            vx = ex2 - ex**2
            stdx = np.sqrt(vx)
            
            c1, c2 = st.columns(2)
            with c1:
                metric_card("Expected Value E(X)", f"{ex:.6f}")
            with c2:
                metric_card("Variance Var(X)", f"{vx:.6f}", f"Std Dev σ: {stdx:.6f}")
                
            sd_prob, _ = quad(f_lamb, max(a_bound, ex - stdx), min(b_bound, ex + stdx))
            st.write(f"**P( |X - E(X)| < σ) (Within 1 SD):** {sd_prob:.6f}")
            st.write(f"**P( |X - E(X)| > σ) (More than 1 SD):** {1 - sd_prob:.6f}")
            
        elif query_mode == "Inverse CDF (Quantile Finder)":
            p_target = st.number_input("Target Probability P(X <= x) = p", min_value=0.0001, max_value=0.9999, value=0.5)
            
            def cdf_val(val):
                res, _ = quad(f_lamb, a_bound, val)
                return res
                
            def root_fun(val):
                return cdf_val(val) - p_target
                
            try:
                x_p = brentq(root_fun, a_bound, b_bound)
                st.success(f"**x value where P(X <= x) = {p_target}:** {x_p:.6f}")
            except Exception as e:
                st.error(f"Could not find root: {e}")
                
        xs = np.linspace(a_bound, b_bound, 500)
        ys = f_lamb(xs)
        ys_cdf = [quad(f_lamb, a_bound, val)[0] for val in xs]
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(xs, ys, color='blue')
        ax[0].set_title("Probability Density Function (PDF)")
        ax[0].fill_between(xs, 0, ys, alpha=0.2, color='blue')
        
        ax[1].plot(xs, ys_cdf, color='red')
        ax[1].set_title("Cumulative Distribution Function (CDF)")
        ax[1].set_ylim(0, 1.1)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error parsing or computing functions: {e}")

# ==========================================
# 14. Joint Discrete PMF
# ==========================================
elif tool == "Joint Discrete PMF":
    st.title("Joint PMF Table Solver")
    st.write("Enter a joint probability mass function table for discrete RVs X and Y.")
    
    num_x = st.number_input("Number of X values", min_value=2, max_value=10, value=3)
    num_y = st.number_input("Number of Y values", min_value=2, max_value=10, value=3)
    
    x_vals_input = st.text_input("X values (comma separated)", "0, 5, 10")
    y_vals_input = st.text_input("Y values (comma separated)", "0, 5, 15")
    
    try:
        x_vals = [float(v.strip()) for v in x_vals_input.split(",")]
        y_vals = [float(v.strip()) for v in y_vals_input.split(",")]
    except:
        x_vals = list(range(num_x))
        y_vals = list(range(num_y))
        
    st.write("Enter joint probabilities p(x, y) row by row (X = rows, Y = columns):")
    joint = np.zeros((len(x_vals), len(y_vals)))
    for i, xv in enumerate(x_vals):
        cols = st.columns(len(y_vals))
        for j, yv in enumerate(y_vals):
            joint[i, j] = cols[j].number_input(f"p({xv},{yv})", value=0.0, min_value=0.0, max_value=1.0, step=0.01, key=f"jp_{i}_{j}")
            
    total = np.sum(joint)
    st.write(f"**Sum of all probabilities: {total:.6f}** (should be ~1.0)")
    
    p_x = np.sum(joint, axis=1)
    p_y = np.sum(joint, axis=0)
    
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    
    ex = np.sum(x_arr * p_x)
    ey = np.sum(y_arr * p_y)
    ex2 = np.sum(x_arr**2 * p_x)
    ey2 = np.sum(y_arr**2 * p_y)
    vx = ex2 - ex**2
    vy = ey2 - ey**2
    
    exy = 0.0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            exy += x_arr[i] * y_arr[j] * joint[i, j]
            
    cov_xy = exy - ex * ey
    cor_xy = cov_xy / (np.sqrt(vx) * np.sqrt(vy)) if vx > 0 and vy > 0 else 0.0
    
    e_sum = ex + ey
    
    e_max = 0.0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            e_max += max(x_arr[i], y_arr[j]) * joint[i, j]
            
    independent = True
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            if not np.isclose(joint[i, j], p_x[i] * p_y[j], atol=1e-6):
                independent = False
                break
                
    df_joint = pd.DataFrame(joint, index=[f"X={v}" for v in x_vals], columns=[f"Y={v}" for v in y_vals])
    df_joint["p_X (marginal)"] = p_x
    margin_row = list(p_y) + [total]
    df_joint.loc["p_Y (marginal)"] = margin_row
    st.dataframe(df_joint)
    
    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.write(f"**E(X) = {ex:.6f}**")
    col1.write(f"**E(Y) = {ey:.6f}**")
    col1.write(f"**E(X+Y) = {e_sum:.6f}**")
    col1.write(f"**E(max(X,Y)) = {e_max:.6f}**")
    col1.write(f"**E(XY) = {exy:.6f}**")
    col2.write(f"**Var(X) = {vx:.6f}**")
    col2.write(f"**Var(Y) = {vy:.6f}**")
    col2.write(f"**Cov(X,Y) = {cov_xy:.6f}**")
    col2.write(f"**Cor(X,Y) = {cor_xy:.6f}**")
    col2.write(f"**Independent? {'Yes' if independent else 'No'}**")
    
    st.subheader("Overflow / Linear Combination")
    st.write("Compute P(aX + bY > c)")
    cc1, cc2, cc3 = st.columns(3)
    a_coef = cc1.number_input("a", value=1.0)
    b_coef = cc2.number_input("b", value=3.0)
    c_thresh = cc3.number_input("c (threshold)", value=5.0)
    p_overflow = 0.0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            if a_coef * x_arr[i] + b_coef * y_arr[j] > c_thresh:
                p_overflow += joint[i, j]
    st.write(f"**P({a_coef}X + {b_coef}Y > {c_thresh}) = {p_overflow:.6f}**")

# ==========================================
# 15. Joint Continuous PDF
# ==========================================
elif tool == "Joint Continuous PDF":
    st.title("Joint Continuous PDF (2D)")
    st.write("Compute probabilities for joint continuous RVs with pdf $f(x, y)$.")
    
    pdf_mode = st.radio("Mode", ["Custom f(x,y)", "Independent Exponentials"])
    
    if pdf_mode == "Custom f(x,y)":
        func_str = st.text_input("f(x, y) =", "x + y")
        col1, col2 = st.columns(2)
        x_lo = col1.number_input("x lower bound", value=0.0)
        x_hi = col1.number_input("x upper bound", value=1.0)
        y_lo = col2.number_input("y lower bound", value=0.0)
        y_hi = col2.number_input("y upper bound", value=1.0)
        
        try:
            x_sym = sp.Symbol('x')
            y_sym = sp.Symbol('y')
            f_sym = sp.sympify(func_str)
            f_lamb = sp.lambdify((y_sym, x_sym), f_sym, 'numpy')
            
            total, _ = dblquad(f_lamb, x_lo, x_hi, y_lo, y_hi)
            st.write(f"**Total integral (should be ~1.0): {total:.6f}**")
            
            st.subheader("Probability Queries")
            q_type = st.selectbox("Query", ["P(X ≤ a ∩ Y ≤ b)", "P(X > a)", "P(X + Y ≤ c)"])
            
            if q_type == "P(X ≤ a ∩ Y ≤ b)":
                c1, c2 = st.columns(2)
                a_val = c1.number_input("a (X upper)", value=1.0)
                b_val = c2.number_input("b (Y upper)", value=1.0)
                p, _ = dblquad(f_lamb, x_lo, min(a_val, x_hi), y_lo, lambda x: min(b_val, y_hi))
                st.write(f"**P(X ≤ {a_val} ∩ Y ≤ {b_val}) = {p:.6f}**")
            elif q_type == "P(X > a)":
                a_val = st.number_input("a", value=3.0)
                p, _ = dblquad(f_lamb, max(a_val, x_lo), x_hi, y_lo, y_hi)
                st.write(f"**P(X > {a_val}) = {p:.6f}**")
            elif q_type == "P(X + Y ≤ c)":
                c_val = st.number_input("c", value=2.0)
                p, _ = dblquad(f_lamb, x_lo, x_hi, y_lo, lambda x: min(c_val - x, y_hi))
                st.write(f"**P(X + Y ≤ {c_val}) = {p:.6f}**")
                
            st.subheader("Moments")
            ex_func = sp.lambdify((y_sym, x_sym), x_sym * f_sym, 'numpy')
            ey_func = sp.lambdify((y_sym, x_sym), y_sym * f_sym, 'numpy')
            ex_val, _ = dblquad(ex_func, x_lo, x_hi, y_lo, y_hi)
            ey_val, _ = dblquad(ey_func, x_lo, x_hi, y_lo, y_hi)
            st.write(f"**E(X) = {ex_val:.6f}**")
            st.write(f"**E(Y) = {ey_val:.6f}**")
            
        except Exception as e:
            st.error(f"Error: {e}")
            
    else:  # Independent Exponentials
        st.write("X, Y independent, each ~ Exp(λ). Joint pdf: f(x,y) = λ²·e^(-λ(x+y)) for x,y ≥ 0.")
        lam_exp = st.number_input("λ", value=1.0, min_value=0.0001)
        
        st.subheader("Probability Queries")
        q_type = st.selectbox("Query", [
            "P(X ≤ a ∩ Y ≤ b)", "P(X + Y ≤ c)", "P(c1 ≤ X+Y ≤ c2)"
        ])
        
        if q_type == "P(X ≤ a ∩ Y ≤ b)":
            c1, c2 = st.columns(2)
            a_val = c1.number_input("a", value=1.0)
            b_val = c2.number_input("b", value=1.0)
            p = (1 - np.exp(-lam_exp * a_val)) * (1 - np.exp(-lam_exp * b_val))
            st.write(f"**P(X ≤ {a_val} ∩ Y ≤ {b_val}) = {p:.6f}**")
        elif q_type == "P(X + Y ≤ c)":
            c_val = st.number_input("c", value=2.0)
            f_exp = lambda y, x: lam_exp**2 * np.exp(-lam_exp*(x+y))
            p, _ = dblquad(f_exp, 0, c_val, 0, lambda x: c_val - x)
            st.write(f"**P(X + Y ≤ {c_val}) = {p:.6f}**")
        elif q_type == "P(c1 ≤ X+Y ≤ c2)":
            c1, c2 = st.columns(2)
            c1_val = c1.number_input("c1 (lower)", value=1.0)
            c2_val = c2.number_input("c2 (upper)", value=2.0)
            f_exp = lambda y, x: lam_exp**2 * np.exp(-lam_exp*(x+y))
            p_upper, _ = dblquad(f_exp, 0, c2_val, 0, lambda x: max(c2_val - x, 0))
            p_lower, _ = dblquad(f_exp, 0, c1_val, 0, lambda x: max(c1_val - x, 0))
            st.write(f"**P({c1_val} ≤ X+Y ≤ {c2_val}) = {p_upper - p_lower:.6f}**")

# ==========================================
# 16. Covariance & Correlation
# ==========================================
elif tool == "Covariance & Correlation":
    st.title("Covariance & Correlation")
    
    cov_mode = st.radio("Input Mode", ["From Raw Paired Data", "Manual Entry (E(X), E(Y), E(XY), Var)"])
    
    if cov_mode == "From Raw Paired Data":
        x_data_input = st.text_area("X data (comma separated)", "1, 2, 3, 4, 5")
        y_data_input = st.text_area("Y data (comma separated)", "2, 4, 5, 4, 5")
        try:
            x_data = np.array([float(v.strip()) for v in x_data_input.split(",")])
            y_data = np.array([float(v.strip()) for v in y_data_input.split(",")])
            if len(x_data) != len(y_data):
                st.error("X and Y must have the same length.")
            else:
                n = len(x_data)
                ddof = st.radio("Denominator", ["Sample (n-1)", "Population (n)"])
                dd = 1 if "Sample" in ddof else 0
                
                ex = np.mean(x_data)
                ey = np.mean(y_data)
                cov = np.cov(x_data, y_data, ddof=dd)[0, 1]
                cor = np.corrcoef(x_data, y_data)[0, 1]
                
                st.write(f"**E(X) = {ex:.6f}**")
                st.write(f"**E(Y) = {ey:.6f}**")
                st.write(f"**Cov(X,Y) = {cov:.6f}**")
                st.write(f"**Cor(X,Y) = {cor:.6f}**")
                
                fig, ax = plt.subplots()
                ax.scatter(x_data, y_data, color='steelblue')
                z = np.polyfit(x_data, y_data, 1)
                p_line = np.poly1d(z)
                ax.plot(np.sort(x_data), p_line(np.sort(x_data)), 'r--')
                ax.set_xlabel("X"); ax.set_ylabel("Y")
                ax.set_title(f"Scatter (r = {cor:.4f})")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
            
    else:
        col1, col2 = st.columns(2)
        ex = col1.number_input("E(X)", value=0.0)
        ey = col1.number_input("E(Y)", value=0.0)
        exy = col2.number_input("E(XY)", value=0.0)
        vx = col1.number_input("Var(X)", value=1.0, min_value=0.0)
        vy = col2.number_input("Var(Y)", value=1.0, min_value=0.0)
        
        cov = exy - ex * ey
        cor = cov / (np.sqrt(vx) * np.sqrt(vy)) if vx > 0 and vy > 0 else 0.0
        
        st.write(f"**Cov(X,Y) = E(XY) − E(X)E(Y) = {exy} − {ex}×{ey} = {cov:.6f}**")
        st.write(f"**Cor(X,Y) = Cov / (σ_X · σ_Y) = {cor:.6f}**")

# ==========================================
# 17. Sampling Distributions & CLT
# ==========================================
elif tool == "Sampling Distributions & CLT":
    st.title("Sampling Distributions & CLT")
    st.write("Explore the distribution of the sample mean $\\bar{X}$.")
    
    mu = st.number_input("Population mean (μ)", value=70.0)
    sigma = st.number_input("Population std dev (σ)", value=1.6, min_value=0.0001)
    
    st.subheader("Compare Sample Sizes")
    sizes_input = st.text_input("Sample sizes to compare (comma separated)", "16, 64")
    try:
        sizes = [int(v.strip()) for v in sizes_input.split(",")]
    except:
        sizes = [16, 64]
        
    results = []
    for n in sizes:
        se = sigma / np.sqrt(n)
        results.append({"n": n, "E(X̄)": mu, "SD(X̄) = σ/√n": se})
    st.dataframe(pd.DataFrame(results))
    
    st.subheader("Query Sample Mean Probabilities")
    query_type = st.selectbox("Query Mode", ["P(X̄ < x)", "P(X̄ > x)", "P(a < X̄ < b)", "P(|X̄ − μ| ≤ d)"])
    
    for n in sizes:
        se = sigma / np.sqrt(n)
        if query_type == "P(X̄ < x)":
            x_target = st.number_input("x", value=mu + 0.2, key=f"x_lt_{n}")
            p = stats.norm.cdf(x_target, mu, se)
            st.write(f"**n = {n}: P(X̄ < {x_target}) = {p:.6f}**")
        elif query_type == "P(X̄ > x)":
            x_target = st.number_input("x", value=mu - 0.2, key=f"x_gt_{n}")
            p = 1 - stats.norm.cdf(x_target, mu, se)
            st.write(f"**n = {n}: P(X̄ > {x_target}) = {p:.6f}**")
        elif query_type == "P(a < X̄ < b)":
            c1, c2 = st.columns(2)
            a_t = c1.number_input("a", value=mu - 0.2, key=f"a_{n}")
            b_t = c2.number_input("b", value=mu + 0.2, key=f"b_{n}")
            p = stats.norm.cdf(b_t, mu, se) - stats.norm.cdf(a_t, mu, se)
            st.write(f"**n = {n}: P({a_t} < X̄ < {b_t}) = {p:.6f}**")
        else:
            d_val = st.number_input("d (deviation from mean)", value=0.5, min_value=0.0, key=f"d_{n}")
            p = stats.norm.cdf(d_val / se) - stats.norm.cdf(-d_val / se)
            st.write(f"**n = {n}: P(|X̄ − {mu}| ≤ {d_val}) = {p:.6f}**")
            
    fig, ax = plt.subplots()
    for n in sizes:
        se = sigma / np.sqrt(n)
        xs = np.linspace(mu - 4*se, mu + 4*se, 300)
        ax.plot(xs, stats.norm.pdf(xs, mu, se), label=f"n={n} (SE={se:.4f})")
    ax.set_title("Sampling Distribution of X̄")
    ax.set_xlabel("X̄")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

# ==========================================
# 18. Method of Moments & MLE
# ==========================================
elif tool == "Method of Moments & MLE":
    st.title("Method of Moments & MLE Solver")
    
    est_type = st.selectbox("Estimator Type", [
        "MoM: Uniform U(0, θ)",
        "MoM/MLE: Exponential",
        "MLE: Poisson",
        "MLE: Normal (both unknown)",
        "MLE: Normal (μ=0, estimate σ²)",
        "Unbiased Estimator Check"
    ])
    
    if est_type == "MoM: Uniform U(0, θ)":
        st.write("For $X \\sim U(0, \\theta)$: $E[X] = \\theta/2$, so $\\hat{\\theta}_{MoM} = 2\\bar{X}$.")
        data_input = st.text_area("Sample data (comma separated)", "1.2, 3.7, 2.1, 4.8, 0.9, 3.3, 2.6, 4.1")
        try:
            data = np.array([float(v.strip()) for v in data_input.split(",")])
            x_bar = np.mean(data)
            theta_hat = 2 * x_bar
            st.write(f"**n = {len(data)}**")
            st.write(f"**X̄ = {x_bar:.6f}**")
            st.write(f"**θ̂_MoM = 2X̄ = {theta_hat:.6f}**")
        except Exception as e:
            st.error(f"Error: {e}")
            
    elif est_type == "MoM/MLE: Exponential":
        st.write("For $X \\sim Exp(\\lambda)$: $E[X] = 1/\\lambda$, so both MoM and MLE lead to $\\hat{\\lambda} = 1/\\bar{X}$.")
        data_input = st.text_area("Sample data (comma separated)", "1.5, 2.3, 0.8, 3.1, 1.2")
        try:
            data = np.array([float(v.strip()) for v in data_input.split(",")])
            x_bar = np.mean(data)
            lam_hat = 1.0 / x_bar
            st.write(f"**n = {len(data)}**")
            st.write(f"**X̄ = {x_bar:.6f}**")
            st.write(f"**λ̂ = 1/X̄ = {lam_hat:.6f}**")
        except Exception as e:
            st.error(f"Error: {e}")
            
    elif est_type == "MLE: Poisson":
        st.write("For Poisson MLE: $\\hat{\\lambda}_{MLE} = \\bar{X}$.")
        data_input = st.text_area("Sample data (comma separated)", "3, 2, 5, 1, 4, 3, 2, 6, 3, 4")
        try:
            data = np.array([float(v.strip()) for v in data_input.split(",")])
            x_bar = np.mean(data)
            n = len(data)
            st.write(f"**λ̂_MLE = X̄ = {x_bar:.6f}**")
            
            lam_range = np.linspace(max(0.1, x_bar - 3), x_bar + 3, 300)
            log_lik = np.array([np.sum(data * np.log(l) - l - np.array([np.log(math.factorial(int(xi))) for xi in data])) for l in lam_range])
            fig, ax = plt.subplots()
            ax.plot(lam_range, log_lik, 'b-')
            ax.axvline(x_bar, color='red', linestyle='--', label=f"λ̂ = {x_bar:.4f}")
            ax.set_xlabel("λ")
            ax.set_ylabel("log L(λ)")
            ax.set_title("Log-Likelihood")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
            
    elif est_type == "MLE: Normal (both unknown)":
        st.write("For $X \\sim N(\\mu, \\sigma^2)$ where both parameters are unknown:")
        st.write("MLE $\\hat{\\mu} = \\bar{X}$")
        st.write("MLE $\\hat{\\sigma}^2 = \\frac{1}{n}\\sum (x_i - \\bar{X})^2$")
        data_input = st.text_area("Sample data (comma separated)", "10.2, 9.8, 11.5, 10.7, 9.1")
        try:
            data = np.array([float(v.strip()) for v in data_input.split(",")])
            n = len(data)
            mu_hat = np.mean(data)
            sigma2_hat = np.var(data, ddof=0)
            st.write(f"**n = {n}**")
            st.write(f"**μ̂ = {mu_hat:.6f}**")
            st.write(f"**σ̂² (MLE) = {sigma2_hat:.6f}**")
            st.write(f"**s² (Unbiased) = {np.var(data, ddof=1):.6f}**")
        except Exception as e:
            st.error(f"Error: {e}")
            
    elif est_type == "MLE: Normal (μ=0, estimate σ²)":
        st.write("For $X \\sim N(0, \\sigma^2)$:")
        st.write("MLE $\\hat{\\sigma}^2 = \\frac{1}{n}\\sum x_i^2$.")
        data_input = st.text_area("Sample data (comma separated)", "0.5, -0.3, 0.8, -1.2, 0.1")
        try:
            data = np.array([float(v.strip()) for v in data_input.split(",")])
            n = len(data)
            sigma2_hat = np.sum(data**2) / n
            st.write(f"**n = {n}**")
            st.write(f"**σ̂² = (1/n)Σxᵢ² = {sigma2_hat:.6f}**")
            st.write(f"**σ̂ = {np.sqrt(sigma2_hat):.6f}**")
        except Exception as e:
            st.error(f"Error: {e}")
            
    elif est_type == "Unbiased Estimator Check":
        st.write("For $f(x) = \\frac{1+\\theta x}{2}$ on $[-1, 1]$, $E[X] = \\theta/3$.")
        st.write("Check which estimator $\\hat{\\theta} = cX$ is unbiased:")
        for c_val, label in [(1, "θ̂ = X̄"), (2, "θ̂ = 2X̄"), (3, "θ̂ = 3X̄")]:
            st.write(f"- **{label}**: E[{label}] = {c_val}·E[X] = {c_val}·(θ/3) = {'θ' if c_val == 3 else f'{c_val}θ/3'}")
        st.success("**θ̂ = 3X̄ is the unbiased estimator** (since E[3X̄] = 3·θ/3 = θ).")

# ==========================================
# 19. Confidence Intervals
# ==========================================
elif tool == "Confidence Intervals":
    st.title("Confidence Intervals")
    
    ci_type = st.selectbox("Interval Type", [
        "Z-interval (σ known)",
        "T-interval (σ unknown)",
        "Proportion Interval (p)",
        "Required Sample Size"
    ])
    
    if ci_type == "Z-interval (σ known)":
        col1, col2 = st.columns(2)
        n = col1.number_input("Sample size (n)", min_value=1, value=100)
        x_bar = col1.number_input("Sample mean (x̄)", value=0.73)
        sigma = col2.number_input("Population σ", value=0.12, min_value=0.0001)
        conf = col2.number_input("Confidence level (%)", value=95.0, min_value=50.0, max_value=99.99)
        
        alpha = 1 - conf / 100
        z_crit = stats.norm.ppf(1 - alpha / 2)
        se = sigma / np.sqrt(n)
        me = z_crit * se
        ci_low = x_bar - me
        ci_high = x_bar + me
        
        st.write(f"**α = {alpha:.4f}, z_{{α/2}} = {z_crit:.4f}**")
        st.write(f"**SE = σ/√n = {se:.6f}**")
        st.write(f"**Margin of Error = {me:.6f}**")
        st.success(f"**{conf}% CI: ({ci_low:.6f}, {ci_high:.6f})**")
        
    elif ci_type == "T-interval (σ unknown)":
        col1, col2 = st.columns(2)
        input_mode = st.radio("Input", ["Raw data", "Summary statistics"])
        
        if input_mode == "Raw data":
            data_input = st.text_area("Data (comma separated)", "52, 58, 61, 55, 49, 63, 57, 54, 60, 53")
            try:
                data = np.array([float(v.strip()) for v in data_input.split(",")])
                n = len(data)
                x_bar = np.mean(data)
                s = np.std(data, ddof=1)
            except:
                st.error("Invalid data")
                n, x_bar, s = 0, 0, 1
        else:
            col1, col2 = st.columns(2)
            n = col1.number_input("n", min_value=2, value=18)
            x_bar = col1.number_input("x̄", value=56.556)
            s = col2.number_input("s (sample std dev)", value=5.34, min_value=0.0001)
            
        conf = st.number_input("Confidence level (%)", value=95.0, min_value=50.0, max_value=99.99)
        
        if n >= 2:
            alpha = 1 - conf / 100
            df = n - 1
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            se = s / np.sqrt(n)
            me = t_crit * se
            ci_low = x_bar - me
            ci_high = x_bar + me
            
            st.write(f"**n = {n}, x̄ = {x_bar:.4f}, s = {s:.4f}**")
            st.write(f"**df = {df}, t_{{α/2}} = {t_crit:.4f}**")
            st.write(f"**SE = s/√n = {se:.6f}**")
            st.write(f"**Margin of Error = {me:.6f}**")
            st.success(f"**{conf}% CI: ({ci_low:.6f}, {ci_high:.6f})**")
            
    elif ci_type == "Proportion Interval (p)":
        col1, col2 = st.columns(2)
        n = col1.number_input("Total sample size n", min_value=1, value=100)
        x_succ = col1.number_input("Number of successes x", min_value=0, max_value=n, value=45)
        conf = col2.number_input("Confidence level (%)", value=95.0, min_value=50.0, max_value=99.99)
        
        p_hat = x_succ / n
        alpha = 1 - conf / 100
        z_crit = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        me = z_crit * se
        ci_low = p_hat - me
        ci_high = p_hat + me
        
        st.write(f"**p̂ = {p_hat:.4f}**")
        st.write(f"**z_{{α/2}} = {z_crit:.4f}**")
        st.write(f"**SE = √[p̂(1-p̂)/n] = {se:.6f}**")
        st.write(f"**Margin of Error = {me:.6f}**")
        st.success(f"**{conf}% CI: ({ci_low:.6f}, {ci_high:.6f})**")
        
    elif ci_type == "Required Sample Size":
        st.write("Find minimum n for a target margin of error.")
        mode = st.radio("Parameter of Interest", ["Mean (continuous)", "Proportion (yes/no)"])
        
        if mode == "Mean (continuous)":
            col1, col2 = st.columns(2)
            sigma = col1.number_input("Population σ", value=0.12, min_value=0.0001)
            me_target = col1.number_input("Target margin of error", value=0.02, min_value=0.0001)
            conf = col2.number_input("Confidence level (%)", value=95.0, min_value=50.0, max_value=99.99)
            
            alpha = 1 - conf / 100
            z_crit = stats.norm.ppf(1 - alpha / 2)
            n_needed = np.ceil((z_crit * sigma / me_target) ** 2)
            st.success(f"**Minimum n = {int(n_needed)}**")
            st.write(f"**(z·σ/ME)² = ({z_crit:.4f} × {sigma} / {me_target})² = {(z_crit * sigma / me_target)**2:.2f}**")
        else:
            col1, col2 = st.columns(2)
            p_guess = col1.number_input("Guessed proportion p (use 0.5 if unknown)", min_value=0.01, max_value=0.99, value=0.5)
            me_target = col1.number_input("Target margin of error", value=0.05, min_value=0.0001)
            conf = col2.number_input("Confidence level (%)", value=95.0, min_value=50.0, max_value=99.99)
            
            alpha = 1 - conf / 100
            z_crit = stats.norm.ppf(1 - alpha / 2)
            n_needed = np.ceil((z_crit**2 * p_guess * (1 - p_guess)) / (me_target**2))
            st.success(f"**Minimum n = {int(n_needed)}**")
            st.write(f"**z²·p(1-p)/ME² = {z_crit:.4f}² × {p_guess} × {1-p_guess} / {me_target}² = {(z_crit**2 * p_guess * (1-p_guess)) / (me_target**2):.2f}**")

# ==========================================
# 20. Z-Tests (1- & 2-sample)
# ==========================================
elif tool == "Z-Tests (1- & 2-sample)":
    st.title("Z-Tests (σ known)")
    
    test_mode = st.radio("Z-Test Type", ["One-sample Z-test", "Two-sample Z-test"])
    
    if test_mode == "One-sample Z-test":
        col1, col2 = st.columns(2)
        x_bar = col1.number_input("Sample mean (x̄)", value=785.0)
        mu0 = col1.number_input("Hypothesised mean (μ₀)", value=800.0)
        sigma = col2.number_input("Population σ", value=40.0, min_value=0.0001)
        n = col2.number_input("Sample size (n)", min_value=1, value=64)
        alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.5)
        tail = st.radio("Alternative hypothesis", ["Two-sided (μ ≠ μ₀)", "Left-tail (μ < μ₀)", "Right-tail (μ > μ₀)"])
        
        se = sigma / np.sqrt(n)
        z_stat = (x_bar - mu0) / se
        
        if "Two-sided" in tail:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif "Left" in tail:
            p_value = stats.norm.cdf(z_stat)
        else:
            p_value = 1 - stats.norm.cdf(z_stat)
            
        reject = p_value < alpha
        
        st.write(f"**SE = σ/√n = {se:.6f}**")
        st.write(f"**z = (x̄ − μ₀) / SE = ({x_bar} − {mu0}) / {se:.4f} = {z_stat:.4f}**")
        st.write(f"**p-value = {p_value:.6f}**")
        if reject:
            st.error(f"**Reject H₀** at α = {alpha} (p = {p_value:.6f} < {alpha})")
        else:
            st.success(f"**Fail to reject H₀** at α = {alpha} (p = {p_value:.6f} ≥ {alpha})")
            
    else:
        st.write("Compare the means of two independent samples when population standard deviations are known.")
        col1, col2 = st.columns(2)
        x1 = col1.number_input("Sample 1 mean (x̄₁)", value=12.5)
        n1 = col1.number_input("Sample 1 size (n₁)", min_value=1, value=35)
        sigma1 = col1.number_input("Population 1 std dev (σ₁)", value=2.1, min_value=0.0001)
        
        x2 = col2.number_input("Sample 2 mean (x̄₂)", value=11.1)
        n2 = col2.number_input("Sample 2 size (n₂)", min_value=1, value=40)
        sigma2 = col2.number_input("Population 2 std dev (σ₂)", value=1.8, min_value=0.0001)
        
        alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.5)
        tail = st.radio("Alternative hypothesis", ["Two-sided (μ₁ ≠ μ₂)", "Left-tail (μ₁ < μ₂)", "Right-tail (μ₁ > μ₂)"])
        
        se = np.sqrt((sigma1**2 / n1) + (sigma2**2 / n2))
        z_stat = (x1 - x2) / se
        
        if "Two-sided" in tail:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif "Left" in tail:
            p_value = stats.norm.cdf(z_stat)
        else:
            p_value = 1 - stats.norm.cdf(z_stat)
            
        reject = p_value < alpha
        
        st.write(f"**SE = √[σ₁²/n₁ + σ₂²/n₂] = {se:.6f}**")
        st.write(f"**z = (x̄₁ − x̄₂) / SE = {z_stat:.4f}**")
        st.write(f"**p-value = {p_value:.6f}**")
        if reject:
            st.error(f"**Reject H₀** at α = {alpha} (p = {p_value:.6f} < {alpha})")
        else:
            st.success(f"**Fail to reject H₀** at α = {alpha} (p = {p_value:.6f} ≥ {alpha})")

# ==========================================
# 21. T-Tests (1-sample, Welch, Paired)
# ==========================================
elif tool == "T-Tests (1-sample, Welch, Paired)":
    st.title("T-Tests")
    
    test_type = st.selectbox("Test Type", [
        "One-sample t-test",
        "Two-sample independent t-test (Welch)",
        "Paired t-test"
    ])
    
    if test_type == "One-sample t-test":
        input_mode = st.radio("Input", ["Raw data", "Summary statistics (x̄, s, n)", "Direct T-statistic (t, n)"])
        
        t_input_mode = False
        if input_mode == "Raw data":
            data_input = st.text_area("Data (comma separated)", "112.3, 97.0, 92.7, 86.0, 102.0, 99.2, 95.8")
            try:
                data = np.array([float(v.strip()) for v in data_input.split(",")])
                n = len(data)
                x_bar = np.mean(data)
                s = np.std(data, ddof=1)
            except:
                st.error("Invalid data")
                n, x_bar, s = 0, 0, 1
        elif input_mode == "Summary statistics (x̄, s, n)":
            col1, col2 = st.columns(2)
            n = col1.number_input("n", min_value=2, value=13)
            x_bar = col1.number_input("x̄", value=96.42)
            s = col2.number_input("s", value=7.5, min_value=0.0001)
        else:
            col1, col2 = st.columns(2)
            n = col1.number_input("n", min_value=2, value=13)
            t_stat_direct = col2.number_input("T-statistic (t)", value=1.6)
            t_input_mode = True
            
        if not t_input_mode:
            mu0 = st.number_input("Hypothesised mean (μ₀)", value=100.0)
            
        alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.5)
        tail = st.radio("Alternative", ["Two-sided (μ ≠ μ₀)", "Left-tail (μ < μ₀)", "Right-tail (μ > μ₀)"])
        
        if n >= 2:
            df = n - 1
            if not t_input_mode:
                se = s / np.sqrt(n)
                t_stat = (x_bar - mu0) / se
            else:
                t_stat = t_stat_direct
                
            if "Two-sided" in tail:
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            elif "Left" in tail:
                p_value = stats.t.cdf(t_stat, df)
            else:
                p_value = 1 - stats.t.cdf(t_stat, df)
                
            reject = p_value < alpha
            
            if not t_input_mode:
                st.write(f"**n = {n}, x̄ = {x_bar:.4f}, s = {s:.4f}**")
                st.write(f"**df = {df}**")
                st.write(f"**t = (x̄ − μ₀) / (s/√n) = {t_stat:.4f}**")
            else:
                st.write(f"**n = {n}, df = {df}**")
                st.write(f"**t = {t_stat:.4f}**")
                
            st.write(f"**p-value = {p_value:.6f}**")
            if reject:
                st.error(f"**Reject H₀** at α = {alpha}")
            else:
                st.success(f"**Fail to reject H₀** at α = {alpha}")
                
    elif test_type == "Two-sample independent t-test (Welch)":
        st.write("Welch's t-test for two independent samples (unequal variances).")
        data1_input = st.text_area("Sample 1 (comma separated)", "1.55, 2.02, 2.02, 2.05, 2.35")
        data2_input = st.text_area("Sample 2 (comma separated)", "1.04, 1.15, 1.23, 1.69, 1.92")
        alpha = st.number_input("Significance level (α)", value=0.01, min_value=0.001, max_value=0.5)
        tail = st.radio("Alternative (μ₁ vs μ₂)", ["Two-sided (μ₁ ≠ μ₂)", "Right-tail (μ₁ > μ₂)", "Left-tail (μ₁ < μ₂)"])
        
        try:
            d1 = np.array([float(v.strip()) for v in data1_input.split(",")])
            d2 = np.array([float(v.strip()) for v in data2_input.split(",")])
            
            n1, n2 = len(d1), len(d2)
            x1, x2 = np.mean(d1), np.mean(d2)
            s1, s2 = np.std(d1, ddof=1), np.std(d2, ddof=1)
            
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            t_stat = (x1 - x2) / se
            
            num = (s1**2/n1 + s2**2/n2)**2
            den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
            df = num / den
            
            if "Two-sided" in tail:
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            elif "Right" in tail:
                p_value = 1 - stats.t.cdf(t_stat, df)
            else:
                p_value = stats.t.cdf(t_stat, df)
                
            reject = p_value < alpha
            
            st.write(f"**Sample 1: n={n1}, x̄={x1:.4f}, s={s1:.4f}**")
            st.write(f"**Sample 2: n={n2}, x̄={x2:.4f}, s={s2:.4f}**")
            st.write(f"**t = {t_stat:.4f}, df = {df:.2f}**")
            st.write(f"**p-value = {p_value:.6f}**")
            if reject:
                st.error(f"**Reject H₀** at α = {alpha}")
            else:
                st.success(f"**Fail to reject H₀** at α = {alpha}")
        except Exception as e:
            st.error(f"Error: {e}")
            
    elif test_type == "Paired t-test":
        st.write("Test whether the mean difference between paired observations differs from Δ₀.")
        data1_input = st.text_area("Sample 1 / Before (comma separated)", "1928, 2549, 2825, 1924, 1628")
        data2_input = st.text_area("Sample 2 / After (comma separated)", "2126, 2885, 2895, 1942, 1750")
        delta0 = st.number_input("Hypothesised mean difference Δ₀ (Sample2 − Sample1)", value=25.0)
        alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.5)
        tail = st.radio("Alternative (μ_d vs Δ₀)", ["Two-sided (μ_d ≠ Δ₀)", "Right-tail (μ_d > Δ₀)", "Left-tail (μ_d < Δ₀)"])
        
        try:
            d1 = np.array([float(v.strip()) for v in data1_input.split(",")])
            d2 = np.array([float(v.strip()) for v in data2_input.split(",")])
            
            if len(d1) != len(d2):
                st.error("Samples must have the same length for a paired test.")
            else:
                diffs = d2 - d1
                n = len(diffs)
                d_bar = np.mean(diffs)
                s_d = np.std(diffs, ddof=1)
                se = s_d / np.sqrt(n)
                t_stat = (d_bar - delta0) / se
                df = n - 1
                
                if "Two-sided" in tail:
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                elif "Right" in tail:
                    p_value = 1 - stats.t.cdf(t_stat, df)
                else:
                    p_value = stats.t.cdf(t_stat, df)
                    
                reject = p_value < alpha
                
                st.write(f"**n = {n}, d̄ = {d_bar:.4f}, s_d = {s_d:.4f}**")
                st.write(f"**t = (d̄ − Δ₀) / (s_d/√n) = {t_stat:.4f}**")
                st.write(f"**df = {df}**")
                st.write(f"**p-value = {p_value:.6f}**")
                if reject:
                    st.error(f"**Reject H₀** at α = {alpha}")
                else:
                    st.success(f"**Fail to reject H₀** at α = {alpha}")
                    
                st.subheader("Paired Differences")
                st.dataframe(pd.DataFrame({"Sample1": d1, "Sample2": d2, "Difference (S2−S1)": diffs}))
        except Exception as e:
            st.error(f"Error: {e}")
