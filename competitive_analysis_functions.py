def competiotn_make_plots([df,pie_frame]):
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "box"}, {"type": "pie"}]],
    )

    fig.add_trace(go.Box( x=df.Metric, 
                         y=df.amount, 
                         boxpoints ="all",
                         hovertext = df["Geography"],
                         jitter =0.5,
                         boxmean = True, 
                         showlegend=False,
                       # hovertemplate = 'Geography: $%{geography}'
                        ), 
                  row=1, col=1)
    #adds a line to reprsentMSRP
    fig.add_trace(go.Scatter(x=['ARP'], y=[get_MSRP(df.loc[0,"UPC"]),get_MSRP(df.loc[0,"UPC"])],
                             mode="lines", name="MSRP", line=dict(color="#000000") ))
    
    #generates pie chart from calculated proportion data
    fig.add_trace(go.Pie(values=pie_frame["count"], labels =pie_frame["bins"], showlegend=True,
                         title_text ="Proportion MSRP Discounts"),
                  row=1, col=2)
    # Update xaxis properties
    fig.update_xaxes(title_text="Distribution of ARP vs Base ARP", row=1, col=1)
    #fig.update(title_text="Proportion of ARP as percentage discount",  row=1, col=2)

    fig.update_yaxes(title_text="Dollars", row=1, col=1)
    #fig.update_yaxes(title_text="Proportion of ARP as percentage discount",  row=1, col=2)

    #label the plot
    fig.update_layout(title_text=df["Item"][0], height=600)
    fig.update_layout(height=600, showlegend=True)

    fig.show()
    #pio.write_html(fig, file=df["Item"][0]+'.html', auto_open=True)

