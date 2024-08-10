"""Generating some plots to get a better intuitive feel for the beta distribution."""
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from scipy import stats

x = np.linspace(0, 1, 200)
data = []
print(
    "the probability that the underlying probability that generated that distribution "
    "is less than 0.5:"
)
for _ in range(5):
    a, b = np.random.randint(0, 30, 2)
    dist = stats.beta(a=a, b=b)
    y = dist.pdf(x)
    df = pd.DataFrame({"x": x, "y": y})
    param = f"a={a},b={b}"
    df["param"] = param
    data.append(
        {
            "a": a,
            "b": b,
            "dist": dist,
            "y": y,
            "df": df,
        }
    )
    print(f"{param} = {dist.cdf(0.5)}")

df = pd.concat([d["df"] for d in data])

fig = px.line(df, "x", "y", color="param", title="random beta distributions")
plot(fig)

# sampling the beta distribution
h = pd.Series(np.random.beta(4, 7, 100_000)).to_frame("h")
fig = px.histogram(h, "h", title="beta sampling distribution")
plot(fig)
