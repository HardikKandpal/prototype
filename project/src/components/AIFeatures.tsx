import React, { useState } from 'react';
import { Search, Brain, DollarSign, BarChart2, Home } from 'lucide-react';
import { PropertyRecommendations } from './PropertyRecommendations';

interface SearchResult {
    id: number;
    title: string;
    location: string;
    city?: string;
    neighborhood?: string;
    price: string;
    total_area: number;
    baths: number;
}

interface PropertySearch {
    location: string;
    min_price?: number;
    max_price?: number;
    bedrooms?: number;
    bathrooms?: number;
    total_area?: number;
    has_balcony?: boolean;
}

interface ComparableProperty {
    id: number;
    title: string;
    location: string;
    price: string;
    total_area: number;
}

interface ApiResponse {
    type: string;
    data: {
        estimated_value?: string;
        min_value?: string;
        max_value?: string;
        confidence_percentage?: number;
        top_features?: Array<{feature: string, importance: number}>;
        comparable_properties?: ComparableProperty[];
        generated_text?: string;
    };
}

export function AIFeatures() {
    // Property Search state
    const [selectedProperty, setSelectedProperty] = useState<number | null>(null);
    const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
    const [searchLoading, setSearchLoading] = useState(false);
    const [searchError, setSearchError] = useState('');
    const [propertySearch, setPropertySearch] = useState<PropertySearch>({
        location: '',
        min_price: undefined,
        max_price: undefined,
        bedrooms: undefined,
        bathrooms: undefined,
        total_area: undefined,
        has_balcony: undefined
    });

    // Description Generator state
    const [propertyDetails, setPropertyDetails] = useState('');
    const [descriptionLoading, setDescriptionLoading] = useState(false);
    const [descriptionError, setDescriptionError] = useState('');
    const [descriptionResult, setDescriptionResult] = useState<string | null>(null);

    // Property Valuation state
    const [location, setLocation] = useState('');
    const [size, setSize] = useState('');
    const [bedrooms, setBedrooms] = useState('');
    const [bathrooms, setBathrooms] = useState('');
    const [hasBalcony, setHasBalcony] = useState(false);
    const [valuationLoading, setValuationLoading] = useState(false);
    const [valuationError, setValuationError] = useState('');
    const [valuationResult, setValuationResult] = useState<ApiResponse['data'] | null>(null);

    const handlePropertySearch = async () => {
        setSearchLoading(true);
        setSearchError('');
        setSearchResults([]);
        
        try {
            // Clean the data before sending
            const cleanedSearch = {
                ...propertySearch,
                min_price: propertySearch.min_price || undefined,
                max_price: propertySearch.max_price || undefined,
                bedrooms: propertySearch.bedrooms || undefined,
                bathrooms: propertySearch.bathrooms || undefined,
                total_area: propertySearch.total_area || undefined
            };

            console.log('Sending search request with params:', cleanedSearch);

            const response = await fetch('http://localhost:5000/api/property-search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cleanedSearch),
            });

            const data = await response.json();
            console.log('Received response:', data);

            if (response.ok) {
                if (data.results && data.results.length > 0) {
                    setSearchResults(data.results);
                    // If we got results but with relaxed filters, show a message
                    if (data.relaxed_search) {
                        setSearchError('Showing results with relaxed filters. Some criteria were ignored to find matches.');
                    }
                } else {
                    setSearchError('No properties found matching your criteria');
                }
            } else {
                setSearchError(data.error || 'Failed to search properties');
            }
        } catch {
            setSearchError('Failed to connect to server');
        } finally {
            setSearchLoading(false);
        }
    };

    const handleGenerateDescription = async () => {
        if (!propertyDetails) {
            setDescriptionError('Please enter property details');
            return;
        }

        setDescriptionLoading(true);
        setDescriptionError('');
        setDescriptionResult(null);

        try {
            const response = await fetch('http://localhost:5000/api/generate-description', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ details: propertyDetails }),
            });

            const data = await response.json();
            if (response.ok) {
                setDescriptionResult(data.generated_text);
            } else {
                setDescriptionError(data.error || 'Something went wrong');
            }
        } catch {
            setDescriptionError('Failed to connect to server');
        } finally {
            setDescriptionLoading(false);
        }
    };

    const handlePropertyValuation = async () => {
        // Validate inputs
        if (!location) {
            setValuationError('Location is required for property valuation');
            return;
        }
        
        if (!size || isNaN(parseInt(size)) || parseInt(size) <= 0) {
            setValuationError('Valid property size is required');
            return;
        }
        
        if (!bedrooms || isNaN(parseInt(bedrooms)) || parseInt(bedrooms) <= 0) {
            setValuationError('Valid number of bedrooms is required');
            return;
        }
        
        if (!bathrooms || isNaN(parseInt(bathrooms)) || parseInt(bathrooms) <= 0) {
            setValuationError('Valid number of bathrooms is required');
            return;
        }
        
        // Prepare data for API call
        const valuationData = {
            location,
            size: parseInt(size),
            bedrooms: parseInt(bedrooms),
            bathrooms: parseInt(bathrooms),
            has_balcony: hasBalcony
        };
        
        setValuationLoading(true);
        setValuationError('');
        setValuationResult(null);

        try {
            const response = await fetch('http://localhost:5000/api/estimate-value', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(valuationData),
            });

            const data = await response.json();
            if (response.ok) {
                setValuationResult(data);
            } else {
                setValuationError(data.error || 'Something went wrong');
            }
        } catch {
            setValuationError('Failed to connect to server');
        } finally {
            setValuationLoading(false);
        }
    };

    return (
        <section className="py-16 bg-gray-100">
            <div className="max-w-6xl mx-auto px-6">
                <h2 className="text-4xl font-bold text-center text-gray-900 mb-12">
                    AI-Powered Real Estate Tools
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Property Search Form */}
                    <div className="bg-white p-6 rounded-lg shadow-lg col-span-2">
                        <h3 className="text-2xl font-semibold flex items-center mb-6">
                            <Search className="h-6 w-6 text-blue-600 mr-2" />
                            Find Your Perfect Property
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="relative">
                                <input
                                    type="text"
                                    value={propertySearch.location}
                                    onChange={(e) => setPropertySearch({
                                        ...propertySearch,
                                        location: e.target.value
                                    })}
                                    placeholder="Location (city, neighborhood, or area)"
                                    className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 w-full"
                                />
                                <div className="text-xs text-gray-500 mt-1">
                                    Enter city name (e.g., "Chennai") or specific area (e.g., "Kanathur, Chennai")
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <input
                                    type="number"
                                    value={propertySearch.min_price || ''}
                                    onChange={(e) => setPropertySearch({
                                        ...propertySearch,
                                        min_price: e.target.value ? Number(e.target.value) : undefined
                                    })}
                                    placeholder="Min Price"
                                    className="w-1/2 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                                />
                                <input
                                    type="number"
                                    value={propertySearch.max_price || ''}
                                    onChange={(e) => setPropertySearch({
                                        ...propertySearch,
                                        max_price: e.target.value ? Number(e.target.value) : undefined
                                    })}
                                    placeholder="Max Price"
                                    className="w-1/2 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                            <input
                                type="number"
                                value={propertySearch.bedrooms || ''}
                                onChange={(e) => setPropertySearch({
                                    ...propertySearch,
                                    bedrooms: e.target.value ? Number(e.target.value) : undefined
                                })}
                                placeholder="Bedrooms"
                                className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                            />
                            <input
                                type="number"
                                value={propertySearch.bathrooms || ''}
                                onChange={(e) => setPropertySearch({
                                    ...propertySearch,
                                    bathrooms: e.target.value ? Number(e.target.value) : undefined
                                })}
                                placeholder="Bathrooms"
                                className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                            />
                            <input
                                type="number"
                                value={propertySearch.total_area || ''}
                                onChange={(e) => setPropertySearch({
                                    ...propertySearch,
                                    total_area: e.target.value ? Number(e.target.value) : undefined
                                })}
                                placeholder="Total Area (sq ft)"
                                className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                            />
                            <div className="flex items-center">
                                <label className="flex items-center space-x-2">
                                    <input
                                        type="checkbox"
                                        checked={propertySearch.has_balcony || false}
                                        onChange={(e) => setPropertySearch({
                                            ...propertySearch,
                                            has_balcony: e.target.checked
                                        })}
                                        className="form-checkbox h-5 w-5 text-blue-600"
                                    />
                                    <span>Has Balcony</span>
                                </label>
                            </div>
                        </div>
                        <button
                            onClick={handlePropertySearch}
                            disabled={searchLoading}
                            className="mt-6 w-full bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition"
                        >
                            {searchLoading ? 'Searching...' : 'Search Properties'}
                        </button>

                        {/* Search Results */}
                        {searchResults.length > 0 && (
                            <div className="mt-6">
                                <h4 className="text-lg font-semibold mb-4">Search Results</h4>
                                
                                {/* Relaxed Search Message */}
                                {searchError && (
                                    <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                                        <p className="text-yellow-700">{searchError}</p>
                                    </div>
                                )}
                                
                                <div className="grid grid-cols-1 gap-4">
                                    {searchResults.map((property) => (
                                        <div 
                                            key={property.id} 
                                            className="p-4 border rounded-lg hover:bg-blue-50 transition cursor-pointer"
                                            onClick={() => setSelectedProperty(property.id)}
                                        >
                                            <h5 className="font-semibold">{property.title}</h5>
                                            <div className="flex justify-between mt-2">
                                                <span className="text-gray-600">{property.location}</span>
                                                <span className="font-medium">{property.price}</span>
                                            </div>
                                            <div className="flex justify-between mt-1 text-sm text-gray-500">
                                                <span>{property.total_area} sq ft</span>
                                                <span>{property.baths} baths</span>
                                            </div>
                                            {property.city && property.neighborhood && (
                                                <div className="mt-1 text-xs text-gray-500">
                                                    {property.neighborhood}, {property.city}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                        
                        {searchResults.length === 0 && searchError && (
                            <div className="mt-6 p-4 border rounded-lg bg-red-50">
                                <p className="text-red-600">{searchError}</p>
                                <p className="text-gray-600 mt-2">
                                    Try broadening your search. For example, just enter "Chennai" instead of a specific neighborhood.
                                </p>
                            </div>
                        )}

                        {/* Property Recommendations */}
                        {selectedProperty !== null && (
                            <div className="mt-8">
                                <PropertyRecommendations 
                                    propertyId={selectedProperty}
                                />
                            </div>
                        )}
                    </div>

                    {/* AI Description Generator */}
                    <div className="bg-white p-6 rounded-lg shadow-lg">
                        <h3 className="text-2xl font-semibold flex items-center mb-6">
                            <Brain className="h-6 w-6 text-blue-600 mr-2" />
                            AI Description Generator
                        </h3>
                        <textarea
                            value={propertyDetails}
                            onChange={(e) => setPropertyDetails(e.target.value)}
                            placeholder="Enter property details (location, size, features, etc.)"
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 h-40"
                        />
                        <button
                            onClick={handleGenerateDescription}
                            className="mt-4 w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 transition"
                            disabled={descriptionLoading || !propertyDetails}
                        >
                            {descriptionLoading ? 'Generating...' : 'Generate Description'}
                        </button>

                        {/* Description Results */}
                        {descriptionResult && (
                            <div className="mt-4 p-4 bg-gray-100 rounded-lg">
                                <h4 className="font-semibold mb-2">Generated Description:</h4>
                                <p className="text-gray-800">{descriptionResult}</p>
                            </div>
                        )}

                        {descriptionError && (
                            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                                <p className="text-red-600">{descriptionError}</p>
                            </div>
                        )}
                    </div>

                    {/* ML-Powered Property Valuation */}
                    <div className="bg-white p-6 rounded-lg shadow-lg">
                        <h3 className="text-2xl font-semibold flex items-center mb-3">
                            <DollarSign className="h-6 w-6 text-blue-600 mr-2" />
                            ML-Powered Property Valuation
                        </h3>
                        <p className="text-gray-600 mb-4">
                            Get an accurate property valuation using our machine learning model.
                        </p>
                        
                        <div className="grid grid-cols-1 gap-4 mb-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                                <input 
                                    type="text" 
                                    value={location} 
                                    onChange={(e) => setLocation(e.target.value)} 
                                    placeholder="Enter property location" 
                                    className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                                />
                                <div className="text-xs text-gray-500 mt-1">
                                    Format: Area, City (e.g., "Adyar, Chennai")
                                </div>
                            </div>
                            
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Total Area (sq ft)</label>
                                <input 
                                    type="number" 
                                    value={size} 
                                    onChange={(e) => setSize(e.target.value)} 
                                    placeholder="Size in sqft" 
                                    className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                            
                            <div className="flex space-x-3">
                                <div className="w-1/2">
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Bedrooms</label>
                                    <input 
                                        type="number" 
                                        value={bedrooms} 
                                        onChange={(e) => setBedrooms(e.target.value)} 
                                        placeholder="Bedrooms" 
                                        className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                                <div className="w-1/2">
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Bathrooms</label>
                                    <input 
                                        type="number" 
                                        value={bathrooms} 
                                        onChange={(e) => setBathrooms(e.target.value)} 
                                        placeholder="Bathrooms" 
                                        className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                            </div>
                            
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Amenities</label>
                                <div className="flex flex-wrap gap-4 mt-2">
                                    <label className="flex items-center space-x-2">
                                        <input
                                            type="checkbox"
                                            checked={hasBalcony}
                                            onChange={(e) => setHasBalcony(e.target.checked)}
                                            className="form-checkbox h-5 w-5 text-blue-600"
                                        />
                                        <span>Balcony</span>
                                    </label>
                                </div>
                            </div>
                        </div>
                        
                        <button 
                            onClick={handlePropertyValuation} 
                            className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 transition" 
                            disabled={valuationLoading}
                        >
                            {valuationLoading ? 'Calculating...' : 'Get Property Valuation'}
                        </button>
                        
                        {/* Valuation Error */}
                        {valuationError && (
                            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                                <p className="text-red-600">{valuationError}</p>
                            </div>
                        )}
                        
                        {/* Valuation Results */}
                        {valuationResult && (
                            <div className="mt-6 bg-gray-50 p-6 rounded-lg border">
                                <h4 className="text-xl font-semibold mb-4">Property Valuation Results</h4>
                                
                                <div className="grid grid-cols-1 gap-4 mb-6">
                                    <div className="bg-white p-4 rounded-lg border shadow-sm">
                                        <p className="text-sm text-gray-500">Estimated Value</p>
                                        <p className="text-2xl font-bold text-blue-600">{valuationResult.estimated_value}</p>
                                    </div>
                                    <div className="bg-white p-4 rounded-lg border shadow-sm">
                                        <p className="text-sm text-gray-500">Value Range</p>
                                        <p className="text-lg font-medium">
                                            {valuationResult.min_value} - {valuationResult.max_value}
                                        </p>
                                    </div>
                                    <div className="bg-white p-4 rounded-lg border shadow-sm">
                                        <p className="text-sm text-gray-500">Confidence</p>
                                        <p className="text-lg font-medium">
                                            ±{valuationResult.confidence_percentage}% margin
                                        </p>
                                    </div>
                                </div>
                                
                                {/* Top Features */}
                                {valuationResult.top_features && valuationResult.top_features.length > 0 && (
                                    <div className="mb-6">
                                        <h5 className="font-semibold text-gray-700 mb-2 flex items-center">
                                            <BarChart2 className="h-5 w-5 mr-1" />
                                            Top Value Factors
                                        </h5>
                                        <div className="bg-white p-3 rounded-lg border">
                                            {valuationResult.top_features.map((feature, index) => (
                                                <div key={index} className="flex justify-between items-center mb-1 last:mb-0">
                                                    <span className="text-gray-700">{feature.feature}</span>
                                                    <div className="w-1/2 bg-gray-200 rounded-full h-2.5">
                                                        <div 
                                                            className="bg-blue-600 h-2.5 rounded-full" 
                                                            style={{ width: `${Math.min(100, feature.importance * 100)}%` }}
                                                        ></div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                
                                {/* Comparable Properties */}
                                {valuationResult.comparable_properties && valuationResult.comparable_properties.length > 0 && (
                                    <div>
                                        <h5 className="font-semibold text-gray-700 mb-2 flex items-center">
                                            <Home className="h-5 w-5 mr-1" />
                                            Comparable Properties
                                        </h5>
                                        <div className="grid grid-cols-1 gap-3">
                                            {valuationResult.comparable_properties.map((property, index) => (
                                                <div key={index} className="bg-white p-3 rounded-lg border flex justify-between items-center">
                                                    <div>
                                                        <p className="font-medium">{property.title}</p>
                                                        <p className="text-sm text-gray-600">{property.location}</p>
                                                        <p className="text-xs text-gray-500">{property.total_area} sq ft</p>
                                                    </div>
                                                    <p className="font-semibold">{property.price}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </section>
    );
}

