$(document).ready(function() {
    let currentPage = 1;
    let currentSymbol = '';
    const newsPerPage = 5;

    // Load companies into dropdown
    $.get('/get_companies', function(data) {
        const select = $('#companySelect');
        data.forEach(company => {
            select.append(new Option(`${company.name} (${company.symbol})`, company.symbol));
        });
    });

    // Function to load sentiment data
    function loadSentimentData(symbol, page = 1) {
        $.ajax({
            url: '/get_sentiment',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ 
                symbol: symbol,
                page: page
            }),
            success: function(response) {
                const table = $('#sentimentTable');
                
                // Clear table if it's the first page
                if (page === 1) {
                    table.empty();
                }

                // Add new rows
                response.news.forEach(item => {
                    const sentimentClass = item.sentiment_score > 0.1 ? 'text-success' : 
                                         item.sentiment_score < -0.1 ? 'text-danger' : 
                                         'text-warning';
                    const sentimentText = item.sentiment_score > 0.1 ? 'Bullish' : 
                                        item.sentiment_score < -0.1 ? 'Bearish' : 
                                        'Neutral';
                    
                    table.append(`
                        <tr>
                            <td>${new Date(item.date).toLocaleDateString()}</td>
                            <td>${item.title}</td>
                            <td class="${sentimentClass}">${sentimentText}</td>
                            <td class="${sentimentClass}">${item.sentiment_score.toFixed(2)}</td>
                        </tr>
                    `);
                });

                // Show/hide load more button
                $('#loadMoreNews').toggle(response.has_more);
            },
            error: function(xhr, status, error) {
                console.error('Error loading sentiment data:', error);
                $('#sentimentTable').html(`
                    <tr>
                        <td colspan="4" class="text-center text-danger">
                            Error loading sentiment data
                        </td>
                    </tr>
                `);
            }
        });
    }

    // Handle company selection change
    $('#companySelect').change(function() {
        const symbol = $(this).val();
        if (symbol) {
            currentSymbol = symbol;
            currentPage = 1;
            loadSentimentData(symbol);
        }
    });

    // Handle load more button click
    $('#loadMoreNews').click(function() {
        currentPage++;
        loadSentimentData(currentSymbol, currentPage);
    });

    // Handle predict button click
    $('#predictBtn').click(function() {
        const symbol = $('#companySelect').val();
        const companyName = $('#companySelect option:selected').text();
        
        if (!symbol) {
            alert('Please select a company first');
            return;
        }
        
        const btn = $(this);
        btn.prop('disabled', true);
        btn.text('Predicting...');
        
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ symbol: symbol }),
            success: function(response) {
                // Update metrics
                $('#mse').text(response.metrics.mse.toFixed(2));
                $('#rmse').text(response.metrics.rmse.toFixed(2));
                $('#mae').text(response.metrics.mae.toFixed(2));
                $('#r2').text(response.metrics.r2.toFixed(4));
                
                // Display the prediction image
                const stockChart = $('#stockChart');
                stockChart.empty();
                stockChart.html(`<img src="${response.plot_url}?t=${new Date().getTime()}" class="img-fluid" alt="Stock Predictions">`);
                
                // Display future predictions
                const futurePredictions = $('#futurePredictions');
                futurePredictions.empty();
                
                if (response.future_predictions && response.future_dates) {
                    futurePredictions.append('<h4>Future Price Predictions (Next 10 Days)</h4>');
                    futurePredictions.append('<table class="table table-striped">');
                    futurePredictions.append('<thead><tr><th>Date</th><th>Predicted Price</th></tr></thead>');
                    futurePredictions.append('<tbody>');
                    
                    response.future_dates.forEach((date, index) => {
                        const price = response.future_predictions[index];
                        futurePredictions.append(`
                            <tr>
                                <td>${new Date(date).toLocaleDateString()}</td>
                                <td>$${price.toFixed(2)}</td>
                            </tr>
                        `);
                    });
                    
                    futurePredictions.append('</tbody></table>');
                }
            },
            error: function(xhr, status, error) {
                alert('Error making prediction: ' + error);
            },
            complete: function() {
                btn.prop('disabled', false);
                btn.text('Predict');
            }
        });
    });
}); 