  var svg = d3.select("svg"),
    margin = {top: 20, right: 20, bottom: 30, left: 40},
    padding=30,
    width = +svg.attr("width") - margin.left - margin.right,
    height = +svg.attr("height") - margin.top - margin.bottom;

  // var x = d3.scaleBand().rangeRound([0, width]).padding(0.1),
  //   y = d3.scaleLinear().rangeRound([height, 0]);
  var g = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    console.log("xVal:",xVal)
    var xScale = d3.scale.ordinal()
				.rangeRoundBands([0, width],.05,0)
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
			.ticks(13);	

	svg.append("g")
			.attr("class", "x axis")
			.attr("transform", "translate(" + 28 + "," + parseInt(parseInt(height) + parseInt(margin.top)+10) + ")");

	svg.selectAll("g.x.axis").transition().duration(500).transition().duration(500).call(xAxis);

	svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
      .attr("transform", "translate(" + 30 + "," + padding + ")")
    .append("text")
      .attr("transform", "rotate(-90)")	
      .attr("y", 24)
      .attr("dy", ".71em")		

	
	svg.selectAll("rect")
   .data(yVal, function(d) { return d; })
   .enter()
   .append("rect")
   .attr("x", function(d, i) {
   		console.log("d here:",d)
   		console.log("padding here:",padding)
		return xScale(xVal[i])+padding;
   })
   .attr("y", function(d) {
   	console.log("d here:")
		return yScale(d)+margin.top;
   })
   .attr("width", xScale.rangeBand())
   .attr("height", function(d) {
		return height -yScale(d);
   })
   .attr("fill", function(d) {
		return "rgb(0, 0, " + 255 + ")";
   })
   .on("mouseover", function(d){
   			d3.select("#tooltip")
					.style("left", d3.event.pageX + "px")
					.style("top", d3.event.pageY + "px")
					.style("opacity", 1)
					.attr("zIndex", 10)
					.select("#value")
					.text(d);
   })
   .on("mouseout", function() {
				d3.select("#tooltip")
					.style("opacity", 0);
	});	