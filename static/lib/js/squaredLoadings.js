  var svg = d3.select("svg"),
    margin = {top: 20, right: 20, bottom: 30, left: 40},
    padding=20,
    width = svg.attr("width") - margin.left - margin.right,
    height = svg.attr("height") - margin.top - margin.bottom;

  // var x = d3.scaleBand().rangeRound([0, width]).padding(0.1),
  //   y = d3.scaleLinear().rangeRound([height, 0]);
  var g = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    console.log("xVal:",xVal)

    var xScale = d3.scale.ordinal()
				.rangeRoundBands([0, width],.2)
				.domain(xVal)

	var yScale = d3.scale.linear()
				.domain([0, d3
					.max(yVal)])
				.range([height,0]);

	xAxis = d3.svg.axis()
			.scale(xScale)
			.orient("bottom")
			.ticks(21);	

	yAxis = d3.svg.axis()
			.scale(yScale)
			.orient("left")
			.ticks(15);	

	svg.append("g")
			.attr("class", "x axis")
			.attr("transform", "translate(" + 35 + "," + parseInt(parseInt(height) + parseInt(margin.top)+10) + ")").append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -2)
      .style("text-anchor", "end")
      .text("Column Names");

	svg.selectAll("g.x.axis").transition().duration(1000).call(xAxis)

	svg.append("g")
      .attr("class", "y axis")
      .attr("transform", "translate(" + 40 + "," + padding + ")")
    .append("text")
      .attr("transform", "rotate(-90)")	
      .attr("y", 24)
      .attr("dy", ".71em")
      .attr("class", "label")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Squared Loadings");	

  svg.selectAll("g.y.axis").transition().duration(1000).call(yAxis);    

  // svg.selectAll("g.y.axis");    	

	
	var bar = svg.selectAll("rect")
   .data(yVal)
   .enter()
   .append("rect")
   .attr("height", function(d) {
    return margin.top;
   })
   .attr("y", function(d) {
    return yScale(0);
   });
   bar.transition()
      .duration(1000)
   .attr("x", function(d, i) {
		return xScale(xVal[i])+45;
   })
   .attr("y", function(d) {
		return yScale(d)+margin.top;
   })
   .attr("width", xScale.rangeBand())
   .attr("height", function(d) {
		return height -yScale(d);
   })
   .attr("fill", function(d) {
		return "rgb(0, 0, " + 255 + ")";
   });
   bar.on("mouseover", function(d){
        d3.select(this)
              .attr("fill", "orange");
   			d3.select("#tooltip")
					.style("left", d3.event.pageX + "px")
					.style("top", d3.event.pageY + "px")
					.style("opacity", 1)
					.attr("zIndex", 10)
					.select("#value")
					.text(d);
   })
   .on("mouseout", function() {
      d3.select(this)
              .attr("fill", function(d) {
                     return "rgb(0, 0, " + 255 + ")";
                   })
				d3.select("#tooltip")
					.style("opacity", 0);
	});	