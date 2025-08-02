# Search with Highlighting in Azure AI Search

## Overview

Search highlighting is a feature that visually emphasizes the search terms within the search results by wrapping them with HTML tags or other markers. This helps users quickly identify why a particular document matched their search query and locate the relevant content within the results.

## How Search Highlighting Works

### Basic Concept
1. **Query Processing**: User submits a search query
2. **Document Matching**: Search engine finds matching documents
3. **Term Identification**: System identifies where query terms appear in the content
4. **Markup Application**: Wraps matching terms with highlight tags
5. **Result Display**: Returns results with highlighted terms for visual emphasis

### Implementation in Our Example

```python
def search_with_highlighting(self, query):
    """Search with result highlighting"""
    try:
        results = self.search_client.search(
            search_text=query,
            highlight_fields="title,content",
            highlight_pre_tag="<mark>",
            highlight_post_tag="</mark>",
            top=5
        )
        
        print(f"\nSearch with highlighting for: '{query}'")
        print("-" * 50)
        
        for result in results:
            print(f"Title: {result.get('title', 'N/A')}")
            
            # Display highlights if available
            if '@search.highlights' in result:
                highlights = result['@search.highlights']
                if 'title' in highlights:
                    print(f"Highlighted Title: {highlights['title'][0]}")
                if 'content' in highlights:
                    print(f"Highlighted Content: {highlights['content'][0]}")
            
            print("-" * 30)
            
    except Exception as e:
        print(f"Error in highlighted search: {e}")
```

## Key Parameters Explained

### 1. `highlight_fields`
```python
highlight_fields="title,content"
```
- **Purpose**: Specifies which fields should have highlighting applied
- **Format**: Comma-separated list of field names
- **Requirement**: Fields must be `SearchableField` type (not `SimpleField`)
- **Example**: Only `title` and `content` fields will show highlights

### 2. `highlight_pre_tag` and `highlight_post_tag`
```python
highlight_pre_tag="<mark>"
highlight_post_tag="</mark>"
```
- **Purpose**: Define the HTML tags to wrap around matching terms
- **Default**: `<em>` and `</em>` if not specified
- **Customization**: Can use any HTML tags or markers
- **Common Options**:
  - `<mark>` and `</mark>` (HTML5 semantic highlighting)
  - `<strong>` and `</strong>` (bold emphasis)
  - `<span class="highlight">` and `</span>` (CSS styling)

### 3. Result Structure
The highlighted results are returned in a special `@search.highlights` property:
```python
if '@search.highlights' in result:
    highlights = result['@search.highlights']
    if 'title' in highlights:
        print(f"Highlighted Title: {highlights['title'][0]}")
```

## Sample Data Context

With our sample documents, here's what highlighting would look like:

### Document Example
```python
{
    "title": "Python Machine Learning Tutorial",
    "content": "Learn how to build machine learning models using Python, scikit-learn, and pandas...",
    "tags": "python, machine learning, tutorial, scikit-learn"
}
```

### Search Examples

#### Query: "machine learning"
**Original Content:**
- Title: "Python Machine Learning Tutorial"
- Content: "Learn how to build machine learning models using Python..."

**Highlighted Results:**
- Highlighted Title: "Python <mark>Machine</mark> <mark>Learning</mark> Tutorial"
- Highlighted Content: "Learn how to build <mark>machine</mark> <mark>learning</mark> models using Python..."

#### Query: "python"
**Original Content:**
- Title: "Python Machine Learning Tutorial"
- Content: "Learn how to build machine learning models using Python, scikit-learn..."

**Highlighted Results:**
- Highlighted Title: "<mark>Python</mark> Machine Learning Tutorial"
- Highlighted Content: "Learn how to build machine learning models using <mark>Python</mark>, scikit-learn..."

## Benefits of Search Highlighting

### For Users
- **Quick Scanning**: Instantly see why a document matched
- **Context Understanding**: Understand the relevance of each result
- **Content Preview**: Get a preview of relevant content without opening documents
- **Improved Navigation**: Find specific information faster

### For Applications
- **Better User Experience**: Users spend less time scanning results
- **Increased Engagement**: Users are more likely to click relevant results
- **Reduced Bounce Rate**: Users find what they're looking for faster
- **Search Quality Feedback**: Shows the effectiveness of search algorithms

## Advanced Highlighting Features

### 1. Multiple Highlight Fragments - Detailed Explanation

Azure AI Search can return multiple highlighted fragments from long content, which is essential for handling large documents effectively.

#### What are Fragments?

**Fragments** are small excerpts of text from a document that contain the search terms, with highlighting applied. Instead of returning the entire field content (which could be very long), Azure AI Search intelligently extracts the most relevant portions.

#### Key Fragment Parameters

```python
def advanced_highlighting(self, query):
    """Advanced highlighting with fragment control"""
    results = self.search_client.search(
        search_text=query,
        highlight_fields="content",
        highlight_pre_tag="<mark>",
        highlight_post_tag="</mark>",
        highlight_fragment_size=150,  # Size of each fragment in characters
        max_highlight_fragments=3     # Maximum fragments per field
    )
    
    for result in results:
        if '@search.highlights' in result:
            highlights = result['@search.highlights']
            if 'content' in highlights:
                print("Content fragments:")
                for i, fragment in enumerate(highlights['content'], 1):
                    print(f"  Fragment {i}: {fragment}")
```

#### Fragment Use Cases

##### Use Case 1: Long Documents with Multiple Matches
**Scenario**: Research papers, technical documentation, or blog posts where search terms appear in multiple sections.

**Example Document**:
```python
{
    "title": "Comprehensive Python Guide",
    "content": """
    Python is a versatile programming language used in web development. 
    [... 2000 characters of content ...]
    Python's machine learning libraries like scikit-learn are powerful.
    [... 1500 characters of content ...]
    Data science with Python involves pandas, numpy, and matplotlib.
    [... 1000 characters of content ...]
    Python web frameworks include Django and Flask for building applications.
    """
}
```

**Query**: "Python machine learning"

**Without Fragments (single long highlight)**:
- Returns entire content with scattered highlights
- Hard to scan for relevant information
- Poor user experience

**With Fragments (fragment_size=150, max_fragments=3)**:
```
Fragment 1: "Python's <mark>machine</mark> <mark>learning</mark> libraries like scikit-learn are powerful for data analysis and model building."
Fragment 2: "Data science with <mark>Python</mark> involves pandas, numpy, and matplotlib for effective analysis."
Fragment 3: "<mark>Python</mark> web frameworks include Django and Flask for building applications."
```

##### Use Case 2: E-commerce Product Descriptions
**Scenario**: Product catalogs with detailed descriptions where users search for specific features.

**Example Product**:
```python
{
    "title": "Professional Camera Kit",
    "description": """
    This professional camera features advanced autofocus technology and 4K video recording.
    The camera body is weather-sealed and includes dual memory card slots.
    [... long technical specifications ...]
    The kit includes multiple lenses: 24-70mm f/2.8, 70-200mm f/4, and 16-35mm f/2.8.
    [... more specifications ...]
    Battery life extends up to 1200 shots per charge with the included lithium battery.
    """
}
```

**Query**: "4K video battery"

**Fragment Results**:
```
Fragment 1: "professional camera features advanced autofocus technology and <mark>4K</mark> <mark>video</mark> recording capabilities"
Fragment 2: "<mark>Battery</mark> life extends up to 1200 shots per charge with the included lithium <mark>battery</mark>"
```

##### Use Case 3: Legal Documents and Contracts
**Scenario**: Large legal documents where users need to find specific clauses or terms.

**Example**: Contract with 50+ pages
**Query**: "termination notice liability"

**Fragment Benefits**:
- Quickly identify relevant sections without reading entire document
- Multiple fragments show different contexts where terms appear
- Legal professionals can assess relevance before deep-diving

##### Use Case 4: News Articles and Blog Posts
**Scenario**: News aggregation or content management systems.

**Example Article**: 3000-word article about technology trends
**Query**: "artificial intelligence privacy"

**Fragment Results Show**:
```
Fragment 1: "<mark>Artificial</mark> <mark>intelligence</mark> systems are transforming healthcare with predictive analytics"
Fragment 2: "Data <mark>privacy</mark> concerns grow as <mark>AI</mark> systems collect more personal information"
Fragment 3: "Companies must balance <mark>AI</mark> innovation with user <mark>privacy</mark> protection"
```

#### Fragment Configuration Strategies

##### 1. Content Type Based Configuration

```python
def content_type_highlighting(self, query, content_type):
    """Configure fragments based on content type"""
    
    if content_type == "short_articles":
        # For blog posts, news articles
        fragment_config = {
            "highlight_fragment_size": 120,
            "max_highlight_fragments": 2
        }
    elif content_type == "technical_docs":
        # For documentation, manuals
        fragment_config = {
            "highlight_fragment_size": 200,
            "max_highlight_fragments": 4
        }
    elif content_type == "product_catalog":
        # For e-commerce descriptions
        fragment_config = {
            "highlight_fragment_size": 100,
            "max_highlight_fragments": 3
        }
    elif content_type == "legal_documents":
        # For contracts, legal texts
        fragment_config = {
            "highlight_fragment_size": 250,
            "max_highlight_fragments": 5
        }
    else:
        # Default configuration
        fragment_config = {
            "highlight_fragment_size": 150,
            "max_highlight_fragments": 3
        }
    
    return self.search_client.search(
        search_text=query,
        highlight_fields="content",
        highlight_pre_tag="<mark>",
        highlight_post_tag="</mark>",
        **fragment_config
    )
```

##### 2. Device-Responsive Fragment Sizing

```python
def responsive_highlighting(self, query, device_type):
    """Adjust fragment size based on device"""
    
    if device_type == "mobile":
        # Smaller fragments for mobile screens
        fragment_size = 80
        max_fragments = 2
    elif device_type == "tablet":
        # Medium fragments for tablets
        fragment_size = 120
        max_fragments = 3
    else:  # desktop
        # Larger fragments for desktop
        fragment_size = 200
        max_fragments = 4
    
    return self.search_client.search(
        search_text=query,
        highlight_fields="content",
        highlight_fragment_size=fragment_size,
        max_highlight_fragments=max_fragments
    )
```

#### Fragment Quality Optimization

##### 1. Smart Fragment Selection
Azure AI Search automatically selects the best fragments based on:
- **Term Density**: Fragments with more search terms get priority
- **Term Proximity**: Fragments where search terms appear close together
- **Document Position**: Earlier fragments may get slight preference

##### 2. Fragment Overlap Handling
```python
# Azure AI Search automatically handles:
# - Avoiding duplicate content in fragments
# - Ensuring fragments don't overlap significantly  
# - Selecting diverse fragments from different document sections
```

#### Advanced Fragment Display Patterns

##### 1. Hierarchical Fragment Display
```python
def hierarchical_fragment_display(self, results):
    """Display fragments with context hierarchy"""
    for result in results:
        print(f"📄 {result['title']}")
        
        if '@search.highlights' in result:
            highlights = result['@search.highlights']
            
            if 'content' in highlights:
                fragments = highlights['content']
                
                if len(fragments) == 1:
                    print(f"   💡 {fragments[0]}")
                else:
                    print(f"   📋 Found in {len(fragments)} sections:")
                    for i, fragment in enumerate(fragments, 1):
                        print(f"   {i}. {fragment}")
                        print("      ...")  # Indicate more content
```

##### 2. Fragment with Context Indicators
```python
def contextual_fragment_display(self, results):
    """Display fragments with position context"""
    for result in results:
        if '@search.highlights' in result:
            highlights = result['@search.highlights']
            
            if 'content' in highlights:
                fragments = highlights['content']
                total_content_length = len(result.get('content', ''))
                
                for i, fragment in enumerate(fragments):
                    # Estimate position in document
                    position_indicator = ""
                    if i == 0:
                        position_indicator = "📍 Beginning"
                    elif i == len(fragments) - 1:
                        position_indicator = "📍 End"
                    else:
                        position_indicator = f"📍 Section {i+1}"
                    
                    print(f"{position_indicator}: {fragment}")
```

#### Performance Implications of Fragments

##### Memory and Processing
- **More Fragments = More Processing**: Each fragment requires text analysis
- **Larger Fragments = More Memory**: Bigger fragments use more memory
- **Optimal Balance**: Usually 2-4 fragments of 100-200 characters each

##### Network Transfer
```python
# Fragment configuration affects response size
# Example response sizes:

# Configuration 1: fragment_size=50, max_fragments=2
# Response: ~100 characters per result

# Configuration 2: fragment_size=200, max_fragments=5  
# Response: ~1000 characters per result

# Choose based on:
# - Network bandwidth constraints
# - Mobile vs desktop users
# - Content preview needs
```

#### Fragment Analytics and Optimization

##### 1. Fragment Effectiveness Tracking
```python
def track_fragment_effectiveness(self, query, results):
    """Track which fragments users find most useful"""
    for result in results:
        if '@search.highlights' in result:
            highlights = result['@search.highlights']
            
            # Log fragment metrics
            fragment_count = len(highlights.get('content', []))
            avg_fragment_length = sum(len(f) for f in highlights.get('content', [])) / max(fragment_count, 1)
            
            # Analytics data
            analytics_data = {
                'query': query,
                'document_id': result['id'],
                'fragment_count': fragment_count,
                'avg_fragment_length': avg_fragment_length,
                'total_highlight_coverage': sum(len(f) for f in highlights.get('content', []))
            }
            
            # Send to analytics system
            log_fragment_analytics(analytics_data)
```

##### 2. A/B Testing Fragment Configurations
```python
def ab_test_fragments(self, query, user_group):
    """A/B test different fragment configurations"""
    
    if user_group == 'A':
        # Configuration A: Fewer, longer fragments
        config = {
            "highlight_fragment_size": 200,
            "max_highlight_fragments": 2
        }
    else:
        # Configuration B: More, shorter fragments  
        config = {
            "highlight_fragment_size": 100,
            "max_highlight_fragments": 4
        }
    
    results = self.search_client.search(
        search_text=query,
        highlight_fields="content",
        **config
    )
    
    # Track user engagement with different configurations
    return results
```

#### Common Fragment Pitfalls and Solutions

##### Pitfall 1: Fragments Too Short
**Problem**: Fragments don't provide enough context
**Solution**: Increase `highlight_fragment_size` to 150-200 characters

##### Pitfall 2: Too Many Fragments
**Problem**: Overwhelming users with too many snippets
**Solution**: Limit to 2-4 fragments per field

##### Pitfall 3: Fragments Miss Important Context
**Problem**: Key information appears outside fragment boundaries
**Solution**: Increase fragment size or use multiple smaller fragments

##### Pitfall 4: Fragment Boundary Issues
**Problem**: Fragments cut off mid-sentence
**Solution**: Azure AI Search automatically handles word boundaries, but consider sentence-aware fragment sizes

### 2. Custom CSS Styling
Instead of `<mark>` tags, you can use CSS classes:

```python
# In your search function
highlight_pre_tag='<span class="search-highlight">'
highlight_post_tag='</span>'
```

```css
/* In your CSS file */
.search-highlight {
    background-color: yellow;
    font-weight: bold;
    padding: 2px 4px;
    border-radius: 3px;
}
```

### 3. Field-Specific Highlighting
You can apply different highlighting to different fields:

```python
def field_specific_highlighting(self, query):
    """Different highlighting for different fields"""
    results = self.search_client.search(
        search_text=query,
        highlight_fields=[
            "title",      # Default highlighting for title
            "content"     # Default highlighting for content
        ],
        highlight_pre_tag="<mark>",
        highlight_post_tag="</mark>"
    )
    
    for result in results:
        if '@search.highlights' in result:
            highlights = result['@search.highlights']
            
            # Title highlighting (usually shorter, more prominent)
            if 'title' in highlights:
                print(f"📝 {highlights['title'][0]}")
            
            # Content highlighting (longer excerpts)
            if 'content' in highlights:
                print(f"💭 ...{highlights['content'][0]}...")
```

## Best Practices

### 1. Field Selection
- **Choose Wisely**: Only highlight fields that users need to see
- **Performance Impact**: More fields = more processing time
- **Relevance**: Highlight fields that explain why document matched

### 2. Fragment Configuration
```python
# Optimal fragment settings
highlight_fragment_size=150    # Good balance of context and conciseness
max_highlight_fragments=2      # Usually sufficient for most use cases
```

### 3. HTML Tag Selection
- **Semantic HTML**: Use `<mark>` for highlighting (HTML5 semantic)
- **Accessibility**: Ensure highlighted content is accessible to screen readers
- **Styling**: Use CSS classes for complex styling needs

### 4. Performance Considerations
- **Field Limitations**: Only highlight searchable fields
- **Index Impact**: Highlighting requires additional processing
- **Caching**: Consider caching highlighted results for common queries

## Common Issues and Solutions

### Issue 1: No Highlights Appearing
**Causes:**
- Field not marked as `SearchableField`
- Query terms not found in specified fields
- Highlighting disabled for certain analyzers

**Solution:**
```python
# Ensure fields are searchable
SearchableField(name="title", type=SearchFieldDataType.String),
SearchableField(name="content", type=SearchFieldDataType.String),
```

### Issue 2: Highlights in Wrong Fields
**Cause:** Including non-searchable fields in `highlight_fields`

**Solution:**
```python
# Only include SearchableField types
highlight_fields="title,content,tags"  # Only if these are SearchableField
```

### Issue 3: Poor Highlight Quality
**Causes:**
- Fragment size too small/large
- Too many/few fragments
- Wrong pre/post tags

**Solution:**
```python
# Adjust fragment parameters
highlight_fragment_size=200,    # Increase for more context
max_highlight_fragments=3,      # Increase for more coverage
```

## Integration Examples

### Web Application Display
```html
<!-- HTML template for displaying highlighted results -->
<div class="search-result">
    <h3 class="result-title">{{ highlighted_title|safe }}</h3>
    <p class="result-snippet">{{ highlighted_content|safe }}</p>
    <div class="result-meta">
        <span>Category: {{ category }}</span>
        <span>Rating: {{ rating }}</span>
    </div>
</div>
```

### JavaScript Processing
```javascript
// Process highlighted results in frontend
function displaySearchResults(results) {
    results.forEach(result => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'search-result';
        
        // Use highlighted title if available, otherwise regular title
        const title = result['@search.highlights']?.title?.[0] || result.title;
        const content = result['@search.highlights']?.content?.[0] || result.content;
        
        resultDiv.innerHTML = `
            <h3>${title}</h3>
            <p>${content}</p>
        `;
        
        document.getElementById('results').appendChild(resultDiv);
    });
}
```

## Testing Highlighting

### Test Cases
1. **Single Term**: "python" → Should highlight "Python" and "python"
2. **Multiple Terms**: "machine learning" → Should highlight both words
3. **Phrase Search**: "\"machine learning\"" → Should highlight exact phrase
4. **Partial Matches**: Depending on analyzer configuration
5. **Case Sensitivity**: Should handle different cases
6. **Special Characters**: Handle punctuation around terms

### Sample Test Function
```python
def test_highlighting():
    """Test various highlighting scenarios"""
    test_queries = [
        "python",
        "machine learning", 
        "data science",
        "javascript web",
        "azure cloud"
    ]
    
    advanced_search = AdvancedSearchExample()
    
    for query in test_queries:
        print(f"\n🧪 Testing: '{query}'")
        advanced_search.search_with_highlighting(query)
        print("-" * 50)
```

## Performance Optimization

### 1. Limit Highlighted Fields
```python
# Instead of highlighting all searchable fields
highlight_fields="title,content"  # Only essential fields
```

### 2. Optimize Fragment Settings
```python
# Balanced settings for performance
highlight_fragment_size=100,      # Smaller fragments = faster processing
max_highlight_fragments=2,        # Fewer fragments = less processing
```

### 3. Cache Common Queries
```python
# Implement caching for popular search terms
def cached_highlighted_search(query):
    cache_key = f"highlight_{query}"
    cached_result = get_from_cache(cache_key)
    
    if cached_result:
        return cached_result
    
    result = search_with_highlighting(query)
    set_cache(cache_key, result, expires_in=300)  # 5 minutes
    return result
```

## Conclusion

Search highlighting is a crucial feature that significantly improves the search experience by helping users understand why documents matched their query and quickly locate relevant information. 

Key takeaways:
- **Implementation**: Simple to implement with Azure AI Search
- **Customization**: Flexible HTML tag and CSS styling options
- **Performance**: Balance between highlighting quality and search speed
- **User Experience**: Dramatically improves result scanning and relevance understanding

The highlighting feature works best when combined with other search features like faceted search and autocomplete to create a comprehensive search experience. Proper configuration of fragment size, field selection, and styling ensures optimal performance and user experience.
