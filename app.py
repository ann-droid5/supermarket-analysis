import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Page Configuration
st.set_page_config(
    page_title="Swift-Cart Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme & Metric Cards
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #1E1E1E;
    }
    
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #2D2D2D;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        border: 1px solid #3E3E3E;
        color: #FAFAFA;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Button Styling */
    .stButton>button {
        color: white;
        background-color: #00ADB5;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00cfd8;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #2D2D2D;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Swift-Cart Strategy Dashboard")
st.markdown("**Strategic Cross-Selling & Inventory Placement Analysis**")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("swiftcart_transactions_P4.csv")
        df['Product_Name'] = df['Product_Name'].str.strip().str.title()
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'swiftcart_transactions_P4.csv' is in the directory.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Data Preprocessing
    transactions = df.groupby('Transaction_ID')['Product_Name'].apply(list).values
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_)

    # Association Rule Mining
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    rules = rules[rules['lift'] > 1.0]

    # Helper to convert frozenset to string
    def frozen_to_str(fset):
        return ', '.join(list(fset))

    rules['antecedents_str'] = rules['antecedents'].apply(frozen_to_str)
    rules['consequents_str'] = rules['consequents'].apply(frozen_to_str)
    rules['rule_desc'] = rules['antecedents_str'] + " → " + rules['consequents_str']

    # Sidebar Navigation
    with st.sidebar:
        st.title("Swift-Cart Analytics")
        st.markdown("---")
        
        menu = st.selectbox(
            "Navigation Module",
            ["Home", 
             "Data Overview", 
             "Frequent Item Analysis", 
             "Association Rules Explorer",
             "Golden Rules",
             "Advanced Visualizations",
             "Business Strategy"],
            index=0
        )
        
        st.markdown("---")
        
        st.caption("Project 4: Market Basket Analysis")

    # --- PAGE LOGIC ---

    if menu == "Home":
        st.subheader("🚀 Project Background")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Objective:** Leverage **Association Rule Mining** to discover hidden product relationships and optimize store layout.
            
            **Business Problems Solved:**
            - **Missed Cross-Selling:** Identifying products often bought together but placed apart.
            - **Inefficient Bundling:** Data-backed promotion strategies.
            - **Web Recommendations:** Improving "Frequently Bought Together" accuracy.
            """)
        with col2:
            st.metric("Total Transactions", f"{df['Transaction_ID'].nunique():,}")
            st.metric("Unique Products", f"{df['Product_Name'].nunique():,}")

    elif menu == "Data Overview":
        st.subheader("📊 Dataset Summary")
        
        # Metric Cards Layout
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions", f"{df['Transaction_ID'].nunique():,}")
        col2.metric("Products", f"{df['Product_Name'].nunique():,}")
        col3.metric("Total Records", f"{df.shape[0]:,}")
        col4.metric("Avg Basket Value", f"${df['Total_Basket_Value'].mean():.2f}")

        st.markdown("### Sample Transactions")
        st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("### Top Selling Products (Raw Count)")
        top_products = df['Product_Name'].value_counts().head(10)
        st.bar_chart(top_products)
        st.caption("This chart shows the absolute number of times each product appears in the purchase records.")

    elif menu == "Frequent Item Analysis":
        st.subheader(" Frequent Itemsets")
        st.markdown("Items that appear most often in transaction baskets.")

        single_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x)==1)].copy()
        single_items['item_name'] = single_items['itemsets'].apply(frozen_to_str)
        single_items = single_items.sort_values(by='support', ascending=False).head(10)

        # Plotting (Dark Theme Compatible)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Set dark background for plot
        fig.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#1E1E1E')
        
        ax.bar(single_items['item_name'], single_items['support'], color='#00ADB5')
        
        # Style axes for visibility on dark background
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')

        plt.xticks(rotation=45)
        plt.xlabel("Product")
        plt.ylabel("Support (Frequency)")
        plt.title("Top 10 Most Frequent Items")
        st.pyplot(fig)
        
        st.info("📝 **Interpretation:** These are your 'anchor' products. High support means they are purchased in a high percentage of all transactions. Ensure these are always in stock.")

    elif menu == "Association Rules Explorer":
        st.subheader("🔍 Product Recommendation Engine")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            product = st.selectbox("Select a Product:", sorted(basket.columns))
            st.markdown(f"**Selected:** {product}")
            st.markdown("Identifying items frequently bought with this product.")
        
        with col2:
            filtered_rules = rules[rules['antecedents'].apply(lambda x: product in x)].copy()
            
            if not filtered_rules.empty:
                top3 = filtered_rules.sort_values(by='lift', ascending=False).head(3)
                
                st.success(f"Top Recommendations for **{product}**:")
                
                for idx, row in top3.iterrows():
                    with st.container():
                        st.markdown(f"### 🔗 Recommend: **{row['consequents_str']}**")
                        # Metric Cards inside recommendation
                        c1, c2 = st.columns(2)
                        c1.metric("Confidence", f"{row['confidence']:.1%}", help="Probability customer creates this purchase")
                        c2.metric("Lift", f"{row['lift']:.2f}x", help="How much more likely than random chance")
                        st.markdown("---")
            else:
                st.warning("No strong association rules found for this product. Try selecting a more common item.")

        st.divider()
        

    elif menu == "Golden Rules":
        st.subheader("🏆 Top 5 Golden Rules")
        st.markdown("The strongest discovered relationships with highest **Lift**.")

        top5 = rules.sort_values(by='lift', ascending=False).head(10)

        for i, row in top5.iterrows():
            with st.expander(f"Rule #{i+1}: {row['antecedents_str']} → {row['consequents_str']}", expanded=True):
                # Metric Cards for Rule details
                col1, col2, col3 = st.columns(3)
                col1.metric("Support", f"{row['support']:.3f}", "Frequency")
                col2.metric("Confidence", f"{row['confidence']:.1%}", "Reliability")
                col3.metric("Lift", f"{row['lift']:.2f}x", "Strength")
                st.write(f"**Insight:** Customers who buy **{row['antecedents_str']}** are **{row['lift']:.2f} times** more likely to buy **{row['consequents_str']}** than the average customer.")

    elif menu == "Advanced Visualizations":
        st.subheader("📊 Advanced Market Basket Analysis")
        st.markdown("Deep dive into association rules using FP-Growth and Network Analysis.")

        # FP-Growth
        st.markdown("### 1. FP-Growth Algorithm Results")
        with st.spinner("Running FP-Growth..."):
             
             frequent_itemsets_fp = fpgrowth(basket, min_support=0.05, use_colnames=True)
             
             # Generate rules from FP-Growth results
             rules_fp = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=1.0)
             rules_sorted_fp = rules_fp.sort_values(by='lift', ascending=False)
        
        st.success("FP-Growth Analysis Complete!")
        st.write(f"Found {len(frequent_itemsets_fp)} frequent itemsets and {len(rules_fp)} association rules.")
        
        st.markdown("#### Top 20 Frequent Itemsets (FP-Growth)")
        
        # Clean up frozenset display
        fp_display = frequent_itemsets_fp.sort_values(by='support', ascending=False).head(20).copy()
        fp_display['itemsets'] = fp_display['itemsets'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(fp_display)

        # 1. Bar Chart: Lift
        st.markdown("### 2. Top 20 Association Rules by Lift")
        top_rules_fp = rules_sorted_fp.head(20)
        
        if not top_rules_fp.empty:
            labels=[
                f"{list(r['antecedents'])[0]} → {list(r['consequents'])[0]}"
                for _ ,r in top_rules_fp.iterrows()
            ]

            fig, ax = plt.subplots(figsize=(10, 5))
            # Transparent background
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')

            plt.barh(labels, top_rules_fp['lift'], color='#00ADB5')
            plt.xlabel("Lift")
            plt.title("Top 10 Association Rules by Lift")
            plt.gca().invert_yaxis() # Invert to show highest at top
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No rules found with sufficient lift.")

        # 2. Network Graph
        st.markdown("### 3. Product Association Network")
        if not top_rules_fp.empty:
            G = nx.DiGraph()
            for _, row in top_rules_fp.iterrows():
                # Handling potentially multiple items in antecedents/consequents by taking the first one
                # as per the user snippet: list(row['...'])[0]
                ant = list(row['antecedents'])[0]
                con = list(row['consequents'])[0]
                G.add_edge(ant, con, weight=row['lift'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            # Transparent background
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            
            pos = nx.spring_layout(G, k=2.5) # k controls spacing
            weights = [G[u][v]['weight'] for u,v in G.edges()]
            
            # Normalize weights for visibility
            width = [w * 1.5 for w in weights]  # Increased width for better visibility

            nx.draw(
                G, pos,
                with_labels=True,
                node_size=3000,
                node_color='#00ADB5',
                edge_color='white', 
                font_color='black', # Black text on nodes might be more readable on teal nodes
                font_weight='bold',
                width=width,
                ax=ax,
                arrows=True,
                arrowstyle='->', 
                arrowsize=20
            )
            plt.title("Product Association Network (Top 20 Rules)", color='white')
            st.pyplot(fig)

        # 3. Heatmap
        st.markdown("### 4. Association Rules Heatmap")
        
        # Prepare data for heatmap
        rules_fp['ant_str'] = rules_fp['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_fp['con_str'] = rules_fp['consequents'].apply(lambda x: ', '.join(list(x)))
        
        top_rules_heatmap = rules_fp.sort_values('lift', ascending=False).head(10)
        
        if not top_rules_heatmap.empty:
             try:
                pivot_data = top_rules_heatmap.pivot_table(
                    index='ant_str',
                    columns='con_str',
                    values='lift',
                    aggfunc='max'
                )
                
                fig, ax = plt.subplots(figsize=(12, 8))
                # Transparent background
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                
                # Heatmap
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    cmap='viridis',
                    fmt='.2f',
                    cbar_kws={'label': 'Lift Score'},
                    ax=ax
                )
                
                # Fix text colors
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(colors='white')
                cbar.set_label('Lift Score', color='white')
                
                ax.tick_params(colors='white')
                plt.xlabel("Consequent (Likely to Buy)", color='white')
                plt.ylabel("Antecedent (If they buy...)", color='white')
                plt.title("Top 10 Association Rules Heatmap", color='white')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                st.pyplot(fig)
             except Exception as e:
                 st.error(f"Could not generate heatmap: {e}")
        else:
            st.warning("Not enough data for heatmap.")

    elif menu == "Business Strategy":
        st.subheader("💡 Strategic Recommendations")
        
        tab1, tab2 = st.tabs(["🏬 Store Layout", "📱 Digital Bundles"])
        
        with tab1:
            st.markdown("### Physical Store Re-Mapping")
            col1, col2 = st.columns(2)
            with col1:
                 st.info("Based on high lift pairs, we recommend placing these items near each other to trigger impulse buys.")
            
            layout_suggestions = rules.sort_values(by='lift', ascending=False).head(1)
            for _, row in layout_suggestions.iterrows():
                 st.success(f"**Action:** Place **{row['consequents_str']}** racks next to the **{row['antecedents_str']}** aisle.")
                 
            st.success("Create a “Breakfast Zone” (Bread, Butter, Coffee, Croissants together)")
            st.success("Place Butter racks adjacent to Bread aisle")
            st.success("Add Coffee displays near bakery section")

        with tab2:
            st.markdown("### Digital \"Recommended for You\" Bundles")
            st.info("Use these combinations for 'Buy Together & Save' app promotions.")
            
            bundle_suggestions = rules.sort_values(by='confidence', ascending=False).head(3)
            
            for _, row in bundle_suggestions.iterrows():
                st.markdown(f"**📦 Bundle Opportunity:** {row['antecedents_str']} + {row['consequents_str']}")
                
else:
    st.warning("Data could not be loaded. Please check the file path.")
