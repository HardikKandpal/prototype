import React, { useEffect, useState } from 'react';

interface Recommendation {
    id: number;
    title: string;
    location: string;
    price: string;
    total_area: number;
    baths: number;
    similarity_score: number;
}

interface PropertyRecommendationsProps {
    propertyId: number;
    maxPrice?: number;
}

export function PropertyRecommendations({ propertyId, maxPrice }: PropertyRecommendationsProps) {
    const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchRecommendations = async () => {
            setLoading(true);
            setError(null);

            try {
                let url = `http://localhost:5000/api/recommendations/${propertyId}`;
                if (maxPrice) {
                    url += `?max_price=${maxPrice}`;
                }

                const response = await fetch(url);
                const data = await response.json();

                if (data.success) {
                    setRecommendations(data.recommendations);
                } else {
                    setError(data.error || 'Failed to fetch recommendations');
                }
            } catch (err) {
                setError('Failed to connect to the server');
            } finally {
                setLoading(false);
            }
        };

        if (propertyId >= 0) {
            fetchRecommendations();
        }
    }, [propertyId, maxPrice]);

    if (loading) {
        return (
            <div className="p-4">
                <div className="animate-pulse flex space-x-4">
                    <div className="flex-1 space-y-4 py-1">
                        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                        <div className="space-y-3">
                            <div className="h-4 bg-gray-200 rounded"></div>
                            <div className="h-4 bg-gray-200 rounded"></div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-4 text-red-600">
                {error}
            </div>
        );
    }

    return (
        <div className="py-8">
            <h2 className="text-2xl font-bold mb-6">Similar Properties</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {recommendations.map((rec) => (
                    <div key={rec.id} className="bg-white rounded-lg shadow-md overflow-hidden">
                        <div className="p-4">
                            <h3 className="text-lg font-semibold mb-2">{rec.title}</h3>
                            <p className="text-gray-600 mb-2">{rec.location}</p>
                            <div className="flex justify-between items-center">
                                <span className="text-blue-600 font-bold">{rec.price}</span>
                                <span className="text-gray-500">{rec.total_area} sq ft</span>
                            </div>
                            <div className="mt-2 text-sm text-gray-500">
                                {rec.baths} Bathrooms
                            </div>
                            <div className="mt-2 text-xs text-gray-400">
                                Similarity: {(rec.similarity_score * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
} 