import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🚀 Simple Streamlit Test")
st.write("This is a basic test to see if Streamlit is working.")

# Simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Simple Sine Wave")

st.pyplot(fig)

if st.button("Click me!"):
    st.write("Button clicked! Streamlit is working! ✅")
    st.balloons()

st.write("If you can see this, Streamlit is running correctly.") 