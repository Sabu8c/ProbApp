import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
try:
    from matplotlib_venn import venn2, venn3
except ImportError:
    pass

st.set_page_config(page_title="MICRO-110 Probability Solver", layout="wide")

st.sidebar.title("MICRO-110 Solver")
st.sidebar.write("Select the problem type to compute:")

module = st.sidebar.selectbox("Problem Type", [
    "1. Frequency & Histograms (Prob 1-3)",
    "2. Descriptive Statistics (Prob 4-5)",
    "3. Set Operations (Prob 6)",
    "4. Sample Space Generator",
    "5. Probability Rules (Prob 7-8)",
    "6. Combinatorics (Prob 9)",
    "7. System Reliability (Prob 10)",
    "8. Binomial Distribution (Prob 11)",
    "9. Continuous Random Variables (Prob 12)",
    "10. Custom Discrete Random Variables",
    "11. Normal Distribution Calculator",
    "12. Poisson Distribution",
    "13. Joint PMF Table",
    "14. Joint Continuous PDF (2D)",
    "15. Covariance & Correlation",
    "16. Sampling Distributions & CLT",
    "17. Method of Moments & MLE",
    "18. Confidence Intervals",
    "19. Z-Test (1-sample)",
    "20. T-Tests (1-sample, 2-sample, paired)"
])

if module == "1. Frequency & Histograms (Prob 1-3)":
    st.header("Frequency & Histograms")
    st.write("Input your data to generate frequency tables and histograms.")
    
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
            st.write(f"**Fraction < {threshold}:** {fraction_less:.4f}")
            st.write(f"**Fraction >= {threshold}:** {fraction_ge:.4f}")

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
                
                df = pd.DataFrame({"Value/Midpoint": vals, "Frequency": freqs, "Relative Freq": rel_freq, "Cumulative Freq": cum_freq})
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

elif module == "2. Descriptive Statistics (Prob 4-5)":
    st.header("Descriptive Statistics")
    
    st.write("Enter multiple datasets to compare them using descriptive statistics and boxplots.")
    num_samples = st.number_input("Number of datasets", min_value=1, max_value=5, value=2)
    
    datasets = []
    labels = []
    
    for i in range(num_samples):
        st.subheader(f"Dataset {i+1}")
        label = st.text_input(f"Label {i+1}", value=f"Sample {i+1}")
        raw_data = st.text_area(f"Data {i+1} (comma separated)", value="1.0, 2.0, 3.0" if i==0 else "2.0, 3.0, 4.0")
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

elif module == "3. Set Operations (Prob 6)":
    st.header("Set Operations & Sample Spaces")
    
    def parse_set(s):
        return set([x.strip() for x in s.split(",") if x.strip()])
        
    s_input = st.text_area("Universal Sample Space S (comma separated):", "1, 2, 3, 4, 5, 6")
    a_input = st.text_input("Event A (comma separated):", "2, 4, 6")
    b_input = st.text_input("Event B (comma separated):", "4, 5, 6")
    
    S = parse_set(s_input)
    A = parse_set(a_input)
    B = parse_set(b_input)
    
    st.write(f"**S:** {S} (Size: {len(S)})")
    st.write(f"**A:** {A}")
    st.write(f"**B:** {B}")
    
    st.write(f"**A ∪ B (Union):** {A.union(B)}")
    st.write(f"**A ∩ B (Intersection):** {A.intersection(B)}")
    st.write(f"**A' (Complement of A):** {S.difference(A)}")
    st.write(f"**B' (Complement of B):** {S.difference(B)}")
    st.write(f"**(A ∪ B)' (Complement of Union):** {S.difference(A.union(B))}")

elif module == "4. Sample Space Generator":
    st.header("Sample Space Generator")
    st.write("Generate and list all exact possible outcomes for your sample space.")
    
    gen_type = st.radio("Generation Type", [
        "Cartesian Product (e.g. flipping multiple coins, rolling dice)", 
        "Combinations (Choosing items, order doesn't matter)", 
        "Permutations (Arranging items, order matters)"
    ])
    
    import itertools
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
        
        # Max value logic: k cannot exceed elements length
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

elif module == "5. Probability Rules (Prob 7-8)":
    st.header("Probability Rules")
    
    scheme = st.radio("Select Scheme", ["Two Events (A, B)", "Three Events (A1, A2, A3)"])
    
    if scheme == "Two Events (A, B)":
        p_a = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.55)
        p_b = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.45)
        p_a_or_b = st.number_input("P(A ∪ B) (at least one)", min_value=0.0, max_value=1.0, value=0.70)
        
        p_a_and_b = p_a + p_b - p_a_or_b
        p_not_a_and_not_b = 1 - p_a_or_b
        
        st.write(f"**P(A ∩ B) (Both):** {p_a_and_b:.4f}")
        st.write(f"**P(A' ∩ B') (Neither / Doesn't consume at least one):** {p_not_a_and_not_b:.4f}")
        if p_a > 0:
            st.write(f"**P(B|A) (B given A):** {p_a_and_b / p_a:.4f}")
        if p_b > 0:
            st.write(f"**P(A|B) (A given B):** {p_a_and_b / p_b:.4f}")
            
        try:
            fig, ax = plt.subplots(figsize=(4, 3))
            v = venn2(subsets=(p_a - p_a_and_b, p_b - p_a_and_b, p_a_and_b), set_labels=('A', 'B'), ax=ax)
            if v and v.get_label_by_id('10'): v.get_label_by_id('10').set_text(f"{p_a - p_a_and_b:.2f}")
            if v and v.get_label_by_id('01'): v.get_label_by_id('01').set_text(f"{p_b - p_a_and_b:.2f}")
            if v and v.get_label_by_id('11'): v.get_label_by_id('11').set_text(f"{p_a_and_b:.2f}")
            st.pyplot(fig)
        except Exception as e:
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
        
        # P(A U B) = P(A) + P(B) - P(AnB) => P(AnB) = P(A) + P(B) - P(AUB)
        p_a1_and_a2 = p_a1 + p_a2 - p_a1_or_a2
        p_a1_and_a3 = p_a1 + p_a3 - p_a1_or_a3
        p_a2_and_a3 = p_a2 + p_a3 - p_a2_or_a3
        
        p_at_least_one = p_a1 + p_a2 + p_a3 - p_a1_and_a2 - p_a1_and_a3 - p_a2_and_a3 + p_all_3
        
        st.write(f"**P(A1') (Not A1):** {1 - p_a1:.4f}")
        st.write(f"**P(A1 ∩ A2) (Both 1 and 2):** {p_a1_and_a2:.4f}")
        st.write(f"**P(A1 ∩ A2 ∩ A3') (Both 1,2 but not 3):** {p_a1_and_a2 - p_all_3:.4f}")
        
        # At most two defects = 1 - P(all three)
        st.write(f"**P(At most two defects):** {1 - p_all_3:.4f}")
        
        st.write(f"**P(A2 | A1):** {p_a1_and_a2 / p_a1:.4f}")
        st.write(f"**P(All Three | A1):** {p_all_3 / p_a1:.4f}")
        
        # Exactly one type = P(A1 u A2 u A3) - P(At least two)
        p_at_least_two = p_a1_and_a2 + p_a1_and_a3 + p_a2_and_a3 - 2 * p_all_3
        p_exactly_one = p_at_least_one - p_at_least_two
        st.write(f"**P(Exactly one | At least one):** {p_exactly_one / p_at_least_one:.4f}")
        st.write(f"**P(Not A3 | A1 ∩ A2):** {(p_a1_and_a2 - p_all_3) / p_a1_and_a2:.4f}")
        
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            v = venn3(subsets=(
                p_a1 - p_a1_and_a2 - p_a1_and_a3 + p_all_3,
                p_a2 - p_a1_and_a2 - p_a2_and_a3 + p_all_3,
                p_a1_and_a2 - p_all_3,
                p_a3 - p_a1_and_a3 - p_a2_and_a3 + p_all_3,
                p_a1_and_a3 - p_all_3,
                p_a2_and_a3 - p_all_3,
                p_all_3
            ), set_labels=('A1', 'A2', 'A3'), ax=ax)
            if v: st.pyplot(fig)
        except Exception:
            pass

elif module == "6. Combinatorics (Prob 9)":
    st.header("Combinatorics")
    
    st.write("Calculate total combinations/runs based on independent parameter categories.")
    
    num_categories = st.number_input("Number of categories (e.g. Temp, Pressure, Catalyst)", min_value=1, value=3)
    counts = []
    for i in range(num_categories):
        counts.append(st.number_input(f"Number of options in category {i+1}", min_value=1, value=[3,4,5][i] if i<3 else 2))
        
    runs = np.prod(counts)
    st.write(f"**Total single-configuration runs possible:** {runs}")
    
    st.write("---")
    st.write("**Permutations & Combinations (nPk / nCk)**")
    n_val = st.number_input("n (total items)", min_value=1, value=10)
    k_val = st.number_input("k (items to select)", min_value=0, max_value=n_val, value=3)
    
    import math
    nCr = math.comb(n_val, k_val)
    nPr = math.perm(n_val, k_val)
    st.write(f"**Combinations ({n_val}C{k_val}):** {nCr}")
    st.write(f"**Permutations ({n_val}P{k_val}):** {nPr}")

elif module == "7. System Reliability (Prob 10)":
    st.header("System Reliability")
    
    st.write("Compute the probability of a system working based on parallel and series circuits.")
    st.write("Enter a boolean probability expression using standard logic operators. Example: `(P1 | P2) & (P3 & P4)`")
    st.write("Where `&` represents series (independent) and `|` represents parallel.")
    
    expression = st.text_input("System Expression", "(P1 | P2) & (P3 & P4)")
    
    st.write("Define Probabilities:")
    col1, col2 = st.columns(2)
    p1 = col1.number_input("P1", value=0.9)
    p2 = col2.number_input("P2", value=0.9)
    p3 = col1.number_input("P3", value=0.8)
    p4 = col2.number_input("P4", value=0.8)
    
    def eval_prob_expr(expr, p_dict):
        # Extremely basic parser for probability boolean algebra
        # We will iterate and replace values, then write a simple recursive solver or use python eval trick
        # A & B => A * B
        # A | B => 1 - (1-A)*(1-B)
        class ProbNode:
            def __init__(self, val):
                self.val = val
            def __and__(self, other):
                return ProbNode(self.val * other.val)
            def __or__(self, other):
                return ProbNode(1 - (1 - self.val) * (1 - other.val))
                
        env = {k: ProbNode(v) for k, v in p_dict.items()}
        try:
            result = eval(expr, {}, env)
            return result.val
        except Exception as e:
            st.error(f"Error evaluating expression: {e}")
            return None
            
    res = eval_prob_expr(expression, {"P1": p1, "P2": p2, "P3": p3, "P4": p4})
    if res is not None:
        st.success(f"**Probability System Works:** {res:.4f}")

elif module == "8. Binomial Distribution (Prob 11)":
    st.header("Binomial Distribution")
    
    n_binom = st.number_input("n (Number of Trials)", min_value=1, value=6)
    p_binom = st.number_input("p (Probability of Success)", min_value=0.0, max_value=1.0, value=0.1)
    
    st.write(f"X ~ Bin({n_binom}, {p_binom})")
    
    x_val = st.number_input("x (Target Successes)", min_value=0, max_value=n_binom, value=1)
    
    prob_exact = stats.binom.pmf(x_val, n_binom, p_binom)
    prob_le = stats.binom.cdf(x_val, n_binom, p_binom)
    prob_ge = 1 - stats.binom.cdf(x_val - 1, n_binom, p_binom) if x_val > 0 else 1.0
    
    st.write(f"**P(X = {x_val}):** {prob_exact:.4f}")
    st.write(f"**P(X <= {x_val}):** {prob_le:.4f}")
    st.write(f"**P(X >= {x_val}):** {prob_ge:.4f}")
    
    mean_v = stats.binom.mean(n_binom, p_binom)
    var_v = stats.binom.var(n_binom, p_binom)
    st.write(f"**Expected Value E(X):** {mean_v}")
    st.write(f"**Variance Var(X):** {var_v}")
    
    # Plotting
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

elif module == "9. Continuous Random Variables (Prob 12)":
    st.header("Continuous Random Variables")
    
    st.write("Input the PDF $f(x)$ as a python/sympy expression (e.g. `1.5*(1-x**2)`). Use `x` as the variable.")
    func_str = st.text_input("f(x) =", "1.5 * (1 - x**2)")
    col1, col2 = st.columns(2)
    a_bound = col1.number_input("Lower Bound (a)", value=-1.0)
    b_bound = col2.number_input("Upper Bound (b)", value=1.0)
    
    try:
        x_sym = sp.Symbol('x')
        f_sym = sp.sympify(func_str)
        
        f_lamb = sp.lambdify(x_sym, f_sym, 'numpy')
        
        # Check validity (integral = 1)
        total_prob, _ = quad(f_lamb, a_bound, b_bound)
        st.write(f"**Total Area (should be ~1.0):** {total_prob:.4f}")
        
        st.markdown("---")
        st.subheader("Compute Probabilities")
        x1 = st.number_input("Lower Limit", value=0.0)
        x2 = st.number_input("Upper Limit", value=0.5)
        
        prob_range, _ = quad(f_lamb, max(a_bound, x1), min(b_bound, x2))
        st.write(f"**P({x1} < X < {x2}):** {prob_range:.4f}")
        
        # Expected Value E(X)
        ex_lamb = sp.lambdify(x_sym, x_sym * f_sym, 'numpy')
        ex, _ = quad(ex_lamb, a_bound, b_bound)
        
        # Variance E(X^2) - E(X)^2
        ex2_lamb = sp.lambdify(x_sym, (x_sym**2) * f_sym, 'numpy')
        ex2, _ = quad(ex2_lamb, a_bound, b_bound)
        vx = ex2 - ex**2
        stdx = np.sqrt(vx)
        
        st.write(f"**Expected Value E(X):** {ex:.4f}")
        st.write(f"**Variance Var(X):** {vx:.4f}")
        st.write(f"**Standard Deviation σ:** {stdx:.4f}")
        
        # P( X within 1 SD of mean )
        sd_prob, _ = quad(f_lamb, max(a_bound, ex - stdx), min(b_bound, ex + stdx))
        st.write(f"**P( |X - E(X)| < σ) (Within 1 SD):** {sd_prob:.4f}")
        st.write(f"**P( |X - E(X)| > σ) (More than 1 SD):** {1 - sd_prob:.4f}")
        
        # Plots
        xs = np.linspace(a_bound, b_bound, 500)
        ys = f_lamb(xs)
        
        # Sympy piecewise CDF logic or numerical CDF
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

elif module == "10. Custom Discrete Random Variables":
    st.header("Custom Discrete Random Variables")
    st.write("Input probabilities for specific discrete outcomes to compute Expected Value and Variance.")
    
    val_input = st.text_input("Values for X (comma separated):", "-1, 0, 1")
    prob_input = st.text_input("Probabilities P(X) (comma separated):", "0.2, 0.5, 0.3")
    
    try:
        x_vals = np.array([float(x.strip()) for x in val_input.split(",")])
        p_vals = np.array([float(x.strip()) for x in prob_input.split(",")])
        
        if len(x_vals) == len(p_vals):
            total_p = np.sum(p_vals)
            st.write(f"**Sum of Probabilities:** {total_p:.4f} (Should normally be ~1.0)")
            
            ex = np.sum(x_vals * p_vals)
            ex2 = np.sum((x_vals**2) * p_vals)
            vx = ex2 - (ex**2)
            stdx = np.sqrt(vx) if vx >= 0 else 0
            
            st.write(f"**Expected Value E(X):** {ex:.4f}")
            st.write(f"**E(X^2):** {ex2:.4f}")
            st.write(f"**Variance Var(X):** {vx:.4f}")
            st.write(f"**Standard Deviation σ:** {stdx:.4f}")
            
            # Simple bar plot
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

elif module == "11. Normal Distribution Calculator":
    st.header("Normal Distribution Calculator")
    st.write("Compute probabilities for $X \\sim N(\\mu, \\sigma^2)$.")

    col1, col2 = st.columns(2)
    mu = col1.number_input("Mean (μ)", value=80.0)
    sigma = col2.number_input("Std Dev (σ)", value=20.0, min_value=0.0001)

    query = st.selectbox("Query Type", [
        "P(X < a)", "P(X > a)", "P(a < X < b)",
        "P(X < a OR X > b)", "P(within k SDs of mean)",
        "Between k1 and k2 SDs from mean",
        "Bolt rejection (count short/long)", "Find x for given cumulative probability"
    ])

    if query == "P(X < a)":
        a = st.number_input("a", value=77.4)
        p = stats.norm.cdf(a, mu, sigma)
        st.write(f"**z = {(a - mu)/sigma:.4f}**")
        st.write(f"**P(X < {a}) = {p:.6f}**")
    elif query == "P(X > a)":
        a = st.number_input("a", value=90.0)
        p = 1 - stats.norm.cdf(a, mu, sigma)
        st.write(f"**z = {(a - mu)/sigma:.4f}**")
        st.write(f"**P(X > {a}) = {p:.6f}**")
    elif query == "P(a < X < b)":
        c1, c2 = st.columns(2)
        a = c1.number_input("a (lower)", value=61.4)
        b = c2.number_input("b (upper)", value=72.9)
        p = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)
        st.write(f"**P({a} < X < {b}) = {p:.6f}**")
    elif query == "P(X < a OR X > b)":
        c1, c2 = st.columns(2)
        a = c1.number_input("a (lower tail)", value=67.6)
        b = c2.number_input("b (upper tail)", value=88.8)
        p = stats.norm.cdf(a, mu, sigma) + (1 - stats.norm.cdf(b, mu, sigma))
        st.write(f"**P(X < {a} or X > {b}) = {p:.6f}**")
    elif query == "P(within k SDs of mean)":
        k = st.number_input("k (number of SDs)", value=1.5, min_value=0.0)
        p = stats.norm.cdf(mu + k*sigma, mu, sigma) - stats.norm.cdf(mu - k*sigma, mu, sigma)
        st.write(f"**P(|X − μ| ≤ {k}σ) = P({mu - k*sigma:.4f} < X < {mu + k*sigma:.4f}) = {p:.6f}**")
        st.write(f"**P(|X − μ| > {k}σ) = {1 - p:.6f}**")
    elif query == "Between k1 and k2 SDs from mean":
        c1, c2 = st.columns(2)
        k1 = c1.number_input("k1 (inner SDs)", value=1.0, min_value=0.0)
        k2 = c2.number_input("k2 (outer SDs)", value=2.0, min_value=0.0)
        p_inner = stats.norm.cdf(mu + k1*sigma, mu, sigma) - stats.norm.cdf(mu - k1*sigma, mu, sigma)
        p_outer = stats.norm.cdf(mu + k2*sigma, mu, sigma) - stats.norm.cdf(mu - k2*sigma, mu, sigma)
        p = p_outer - p_inner
        st.write(f"**P({k1}σ < |X − μ| < {k2}σ) = {p:.6f}**")
    elif query == "Bolt rejection (count short/long)":
        st.write("Given N items from N(μ,σ), count expected rejects outside [lower, upper].")
        c1, c2, c3 = st.columns(3)
        lower = c1.number_input("Lower limit", value=2.9)
        upper = c2.number_input("Upper limit", value=3.1)
        N = c3.number_input("Total items N", value=397, min_value=1)
        p_short = stats.norm.cdf(lower, mu, sigma)
        p_long = 1 - stats.norm.cdf(upper, mu, sigma)
        st.write(f"**P(X < {lower}) = {p_short:.6f}  →  n_short ≈ {p_short * N:.1f}**")
        st.write(f"**P(X > {upper}) = {p_long:.6f}  →  n_long ≈ {p_long * N:.1f}**")
        st.write(f"**Expected accepted: {N - p_short*N - p_long*N:.1f}**")
    elif query == "Find x for given cumulative probability":
        p_target = st.number_input("Cumulative probability P(X ≤ x)", value=0.95, min_value=0.0001, max_value=0.9999)
        x_val = stats.norm.ppf(p_target, mu, sigma)
        st.write(f"**x = {x_val:.6f}**")

    # Plot
    xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    ys = stats.norm.pdf(xs, mu, sigma)
    fig, ax = plt.subplots()
    ax.plot(xs, ys, 'b-')
    ax.fill_between(xs, 0, ys, alpha=0.1, color='blue')
    ax.set_title(f"N({mu}, {sigma}²)")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    st.pyplot(fig)

elif module == "12. Poisson Distribution":
    st.header("Poisson Distribution")
    st.write("Compute probabilities for $X \\sim \\text{Pois}(\\lambda)$.")

    lam_base = st.number_input("Base rate λ (per unit)", value=10.0, min_value=0.0001)
    volume = st.number_input("Volume / time multiplier", value=1.0, min_value=0.0001)
    lam = lam_base * volume
    st.write(f"**Effective λ = {lam_base} × {volume} = {lam:.4f}**")

    query_p = st.selectbox("Query", [
        "P(X = k)", "P(X ≤ k)", "P(X ≥ k)",
        "P(X > μ + 1·SD)", "Find volume for P(X ≥ 1) = target"
    ])

    if query_p == "P(X = k)":
        k = st.number_input("k", value=8, min_value=0)
        st.write(f"**P(X = {k}) = {stats.poisson.pmf(k, lam):.6f}**")
    elif query_p == "P(X ≤ k)":
        k = st.number_input("k", value=7, min_value=0)
        st.write(f"**P(X ≤ {k}) = {stats.poisson.cdf(k, lam):.6f}**")
    elif query_p == "P(X ≥ k)":
        k = st.number_input("k", value=8, min_value=0)
        p = 1 - stats.poisson.cdf(k - 1, lam) if k > 0 else 1.0
        st.write(f"**P(X ≥ {k}) = {p:.6f}**")
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
        # P(X>=1) = 1 - e^(-λ*v) = target  =>  v = -ln(1-target)/λ_base
        v_needed = -np.log(1 - target) / lam_base
        st.write(f"**Required volume = {v_needed:.6f}**")

    st.write(f"**E(X) = {lam:.4f},  Var(X) = {lam:.4f},  SD = {np.sqrt(lam):.4f}**")

    xs_p = np.arange(0, int(lam + 4*np.sqrt(lam)) + 1)
    fig, ax = plt.subplots()
    ax.bar(xs_p, stats.poisson.pmf(xs_p, lam), color='lightblue', edgecolor='black')
    ax.set_title(f"Poisson(λ={lam:.2f}) PMF"); ax.set_xlabel("k"); ax.set_ylabel("P(X=k)")
    st.pyplot(fig)

elif module == "13. Joint PMF Table":
    st.header("Joint PMF Table Solver")
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

    # Marginals
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

    # E(XY)
    exy = 0.0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            exy += x_arr[i] * y_arr[j] * joint[i, j]

    cov_xy = exy - ex * ey
    cor_xy = cov_xy / (np.sqrt(vx) * np.sqrt(vy)) if vx > 0 and vy > 0 else 0.0

    # E(X+Y)
    e_sum = ex + ey

    # E(max(X,Y))
    e_max = 0.0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            e_max += max(x_arr[i], y_arr[j]) * joint[i, j]

    # Independence check
    independent = True
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            if not np.isclose(joint[i, j], p_x[i] * p_y[j], atol=1e-6):
                independent = False
                break

    # Display table with marginals
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

    st.subheader("CDF Queries")
    st.write("Compute probabilities from the CDF: P(X = k) = F(k) − F(k−1)")
    cdf_x_vals = np.sort(np.unique(x_arr))
    cdf_pmf = {}
    for xv in cdf_x_vals:
        idx = np.where(x_arr == xv)[0][0]
        cdf_pmf[xv] = p_x[idx]
    cdf_vals = {}
    running = 0.0
    for xv in cdf_x_vals:
        running += cdf_pmf[xv]
        cdf_vals[xv] = running
    st.write("**CDF F(x):**", {f"{k}": f"{v:.4f}" for k, v in cdf_vals.items()})

elif module == "14. Joint Continuous PDF (2D)":
    st.header("Joint Continuous PDF (2D)")
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
            from scipy.integrate import dblquad
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

            # Marginals and moments
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
            from scipy.integrate import dblquad
            f_exp = lambda y, x: lam_exp**2 * np.exp(-lam_exp*(x+y))
            p, _ = dblquad(f_exp, 0, c_val, 0, lambda x: c_val - x)
            st.write(f"**P(X + Y ≤ {c_val}) = {p:.6f}**")
        elif q_type == "P(c1 ≤ X+Y ≤ c2)":
            c1, c2 = st.columns(2)
            c1_val = c1.number_input("c1 (lower)", value=1.0)
            c2_val = c2.number_input("c2 (upper)", value=2.0)
            from scipy.integrate import dblquad
            f_exp = lambda y, x: lam_exp**2 * np.exp(-lam_exp*(x+y))
            p_upper, _ = dblquad(f_exp, 0, c2_val, 0, lambda x: max(c2_val - x, 0))
            p_lower, _ = dblquad(f_exp, 0, c1_val, 0, lambda x: max(c1_val - x, 0))
            st.write(f"**P({c1_val} ≤ X+Y ≤ {c2_val}) = {p_upper - p_lower:.6f}**")

elif module == "15. Covariance & Correlation":
    st.header("Covariance & Correlation")

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

elif module == "16. Sampling Distributions & CLT":
    st.header("Sampling Distributions & CLT")
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

    st.subheader("P(|X̄ − μ| ≤ d)")
    d_val = st.number_input("d (deviation from mean)", value=1.0, min_value=0.0)
    for n in sizes:
        se = sigma / np.sqrt(n)
        p = stats.norm.cdf(d_val / se) - stats.norm.cdf(-d_val / se)
        st.write(f"**n = {n}: P(|X̄ − {mu}| ≤ {d_val}) = {p:.6f}**")

    # Plot comparison
    fig, ax = plt.subplots()
    for n in sizes:
        se = sigma / np.sqrt(n)
        xs = np.linspace(mu - 4*se, mu + 4*se, 300)
        ax.plot(xs, stats.norm.pdf(xs, mu, se), label=f"n={n} (SE={se:.4f})")
    ax.set_title("Sampling Distribution of X̄")
    ax.set_xlabel("X̄"); ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

elif module == "17. Method of Moments & MLE":
    st.header("Method of Moments & MLE")

    est_type = st.selectbox("Estimator Type", [
        "MoM: Uniform U(0, θ)",
        "MoM: Poisson",
        "MLE: Poisson",
        "MLE: Normal (μ=0, estimate σ²)",
        "Unbiased Estimator Check: f(x) = (1+θx)/2"
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

    elif est_type == "MoM: Poisson":
        st.write("For $X \\sim \\text{Pois}(\\lambda)$: $E[X] = \\lambda$, so $\\hat{\\lambda}_{MoM} = \\bar{X}$.")
        data_input = st.text_area("Sample data (comma separated)", "3, 2, 5, 1, 4, 3, 2, 6, 3, 4")
        true_lam = st.number_input("True λ (for comparison, optional)", value=3.2)
        try:
            data = np.array([float(v.strip()) for v in data_input.split(",")])
            x_bar = np.mean(data)
            st.write(f"**n = {len(data)}**")
            st.write(f"**λ̂_MoM = X̄ = {x_bar:.6f}**")
            st.write(f"**True λ = {true_lam}**")
            st.write(f"**Absolute error = {abs(x_bar - true_lam):.6f}**")

            # PMF comparison
            k_max = int(max(data.max(), x_bar + 3*np.sqrt(x_bar))) + 1
            ks = np.arange(0, k_max + 1)
            empirical_pmf = np.array([np.sum(data == k) / len(data) for k in ks])
            theoretical_pmf = stats.poisson.pmf(ks, x_bar)
            fig, ax = plt.subplots()
            w = 0.35
            ax.bar(ks - w/2, empirical_pmf, w, label="Empirical", color='steelblue')
            ax.bar(ks + w/2, theoretical_pmf, w, label=f"Pois(λ̂={x_bar:.2f})", color='coral')
            ax.legend(); ax.set_xlabel("k"); ax.set_ylabel("P(X=k)")
            ax.set_title("Empirical vs Theoretical PMF")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

    elif est_type == "MLE: Poisson":
        st.write("For Poisson MLE: $\\hat{\\lambda}_{MLE} = \\bar{X}$ (same as MoM).")
        data_input = st.text_area("Sample data (comma separated)", "3, 2, 5, 1, 4, 3, 2, 6, 3, 4")
        try:
            data = np.array([float(v.strip()) for v in data_input.split(",")])
            x_bar = np.mean(data)
            n = len(data)
            st.write(f"**λ̂_MLE = X̄ = {x_bar:.6f}**")

            # Log-likelihood surface
            lam_range = np.linspace(max(0.1, x_bar - 3), x_bar + 3, 300)
            log_lik = np.array([np.sum(data * np.log(l) - l - np.array([np.log(np.math.factorial(int(xi))) for xi in data])) for l in lam_range])
            fig, ax = plt.subplots()
            ax.plot(lam_range, log_lik, 'b-')
            ax.axvline(x_bar, color='red', linestyle='--', label=f"λ̂ = {x_bar:.4f}")
            ax.set_xlabel("λ"); ax.set_ylabel("log L(λ)")
            ax.set_title("Log-Likelihood"); ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

    elif est_type == "MLE: Normal (μ=0, estimate σ²)":
        st.write("For $X \\sim N(0, \\sigma^2)$: MLE $\\hat{\\sigma}^2 = \\frac{1}{n}\\sum x_i^2$.")
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

    elif est_type == "Unbiased Estimator Check: f(x) = (1+θx)/2":
        st.write("For $f(x) = \\frac{1+\\theta x}{2}$ on $[-1, 1]$, $E[X] = \\theta/3$.")
        st.write("Check which estimator $\\hat{\\theta} = cX$ is unbiased:")
        for c_val, label in [(1, "θ̂ = X̄"), (2, "θ̂ = 2X̄"), (3, "θ̂ = 3X̄")]:
            st.write(f"- **{label}**: E[{label}] = {c_val}·E[X] = {c_val}·(θ/3) = {'θ' if c_val == 3 else f'{c_val}θ/3'}")
        st.success("**θ̂ = 3X̄ is the unbiased estimator** (since E[3X̄] = 3·θ/3 = θ).")

elif module == "18. Confidence Intervals":
    st.header("Confidence Intervals")

    ci_type = st.selectbox("Interval Type", [
        "Z-interval (σ known)",
        "T-interval (σ unknown)",
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
            data_input = st.text_area("Data (comma separated)", "52, 58, 61, 55, 49, 63, 57, 54, 60, 53, 66, 50, 62, 56, 48, 59, 51, 64")
            try:
                data = np.array([float(v.strip()) for v in data_input.split(",")])
                n = len(data)
                x_bar = np.mean(data)
                s = np.std(data, ddof=1)
            except:
                st.error("Invalid data"); n = 0; x_bar = 0; s = 1
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

    elif ci_type == "Required Sample Size":
        st.write("Find minimum n for a given margin of error (z-interval).")
        col1, col2 = st.columns(2)
        sigma = col1.number_input("Population σ", value=0.12, min_value=0.0001)
        me_target = col1.number_input("Target margin of error", value=0.02, min_value=0.0001)
        conf = col2.number_input("Confidence level (%)", value=95.0, min_value=50.0, max_value=99.99)

        alpha = 1 - conf / 100
        z_crit = stats.norm.ppf(1 - alpha / 2)
        n_needed = np.ceil((z_crit * sigma / me_target) ** 2)
        st.success(f"**Minimum n = {int(n_needed)}**")
        st.write(f"**(z·σ/ME)² = ({z_crit:.4f} × {sigma} / {me_target})² = {(z_crit * sigma / me_target)**2:.2f}**")

elif module == "19. Z-Test (1-sample)":
    st.header("Z-Test (1-sample, σ known)")

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

elif module == "20. T-Tests (1-sample, 2-sample, paired)":
    st.header("T-Tests")

    test_type = st.selectbox("Test Type", [
        "One-sample t-test",
        "Two-sample independent t-test (Welch)",
        "Paired t-test"
    ])

    if test_type == "One-sample t-test":
        input_mode = st.radio("Input", ["Raw data", "Summary statistics (x̄, s, n)", "Direct T-statistic (t, n)"])

        t_input_mode = False
        if input_mode == "Raw data":
            data_input = st.text_area("Data (comma separated)", "112.3, 97.0, 92.7, 86.0, 102.0, 99.2, 95.8, 103.5, 89.0, 86.7")
            try:
                data = np.array([float(v.strip()) for v in data_input.split(",")])
                n = len(data)
                x_bar = np.mean(data)
                s = np.std(data, ddof=1)
            except:
                st.error("Invalid data"); n = 0; x_bar = 0; s = 1
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
        data1_input = st.text_area("Sample 1 (comma separated)", "1.55, 2.02, 2.02, 2.05, 2.35, 2.57, 2.93, 2.94, 2.97")
        data2_input = st.text_area("Sample 2 (comma separated)", "1.04, 1.15, 1.23, 1.69, 1.92, 1.98, 2.36, 2.49, 2.72, 1.37, 1.43, 1.57, 1.71, 1.94, 2.06, 2.55, 2.64, 2.82")
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

            # Welch df
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
        data1_input = st.text_area("Sample 1 / Before (comma separated)", "1928, 2549, 2825, 1924, 1628, 2175, 2114, 2621, 1843, 2541")
        data2_input = st.text_area("Sample 2 / After (comma separated)", "2126, 2885, 2895, 1942, 1750, 2184, 2164, 2626, 2006, 2627")
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
                st.write(f"**t = (d̄ − Δ₀) / (s_d/√n) = ({d_bar:.4f} − {delta0}) / {se:.4f} = {t_stat:.4f}**")
                st.write(f"**df = {df}**")
                st.write(f"**p-value = {p_value:.6f}**")
                if reject:
                    st.error(f"**Reject H₀** at α = {alpha}")
                else:
                    st.success(f"**Fail to reject H₀** at α = {alpha}")

                # Show differences
                st.subheader("Paired Differences")
                st.dataframe(pd.DataFrame({"Sample1": d1, "Sample2": d2, "Difference (S2−S1)": diffs}))
        except Exception as e:
            st.error(f"Error: {e}")

